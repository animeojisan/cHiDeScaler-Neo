// DirectML buffer -> D3D12 compute -> shared D3D11 texture. No CPU frame pixels.
#include <windows.h>
#include <d3d11_1.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <mutex>
#include <vector>
using Microsoft::WRL::ComPtr;
static thread_local char bridge_error[256]={};
static void check(HRESULT hr,const char*what){if(FAILED(hr)){char b[128];sprintf_s(b,"%s: HRESULT=0x%08lx",what,(unsigned long)hr);throw std::runtime_error(b);}}
class DmlTexture {
 ComPtr<ID3D12Resource> inflight_source_;ComPtr<ID3D12Device> device_;ComPtr<ID3D11Device> device11_;ComPtr<ID3D11Texture2D> image11_,display11_;ComPtr<ID3D11DeviceContext> context11_;ComPtr<ID3D12Resource> image_;
 ComPtr<ID3D12CommandQueue> queue_;ComPtr<ID3D12CommandAllocator> allocator_;ComPtr<ID3D12GraphicsCommandList> list_;ComPtr<ID3D12Fence> fence_;
 std::vector<ComPtr<ID3D12Resource>> inflight_inputs_;ComPtr<ID3D12RootSignature> interp_root_;ComPtr<ID3D12PipelineState> interp_pipeline_;
 ComPtr<ID3D12RootSignature> pack_root_;ComPtr<ID3D12PipelineState> pack_pipeline_;ComPtr<ID3D11Query> pack_query_;
 ComPtr<ID3D12RootSignature> root_;ComPtr<ID3D12PipelineState> pipeline_;ComPtr<ID3D12DescriptorHeap> heap_;UINT width_,height_;UINT64 value_=0;HANDLE event_=nullptr;
 static D3D12_RESOURCE_BARRIER barrier(ID3D12Resource*r,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){D3D12_RESOURCE_BARRIER b={};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};return b;}
public:
 DmlTexture(ID3D12Device*device,UINT w,UINT h):device_(device),width_(w),height_(h){
  if(!w||!h||w>8192||h>8192)throw std::runtime_error("invalid texture extent");
  ComPtr<IDXGIFactory4> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");ComPtr<IDXGIAdapter> adapter;check(factory->EnumAdapterByLuid(device->GetAdapterLuid(),IID_PPV_ARGS(&adapter)),"match DML adapter");
  check(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device11_,nullptr,&context11_),"D3D11 sharing device");
  D3D12_HEAP_PROPERTIES hp={};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC rd={};rd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;rd.Width=w;rd.Height=h;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.Format=DXGI_FORMAT_R8G8B8A8_UNORM;rd.SampleDesc.Count=1;rd.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS|D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET|D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
  check(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_SHARED,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&image_)),"shared RGBA texture");
  HANDLE shared=nullptr;check(device->CreateSharedHandle(image_.Get(),nullptr,GENERIC_ALL,nullptr,&shared),"share RGBA handle");ComPtr<ID3D11Device1>d1;HRESULT status=device11_.As(&d1);if(SUCCEEDED(status))status=d1->OpenSharedResource1(shared,IID_PPV_ARGS(&image11_));CloseHandle(shared);check(status,"D3D11 import RGBA texture");
  // Intel WGL rejects the directly imported D3D12-origin texture. Keep an
  // ordinary D3D11-owned surface as WGC's validated interop direction does.
  D3D11_TEXTURE2D_DESC owned={};image11_->GetDesc(&owned);owned.MiscFlags=0;owned.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;owned.CPUAccessFlags=0;owned.Usage=D3D11_USAGE_DEFAULT;check(device11_->CreateTexture2D(&owned,nullptr,&display11_),"D3D11 owned RGBA surface");

  D3D12_COMMAND_QUEUE_DESC q={};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue_)),"convert queue");check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator_)),"convert allocator");check(device->CreateCommandList(0,q.Type,allocator_.Get(),nullptr,IID_PPV_ARGS(&list_)),"convert command list");check(list_->Close(),"close convert list");check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)),"convert fence");
  D3D12_DESCRIPTOR_HEAP_DESC hd={};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=2;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap_)),"RGBA descriptor heap");
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav={};uav.Format=rd.Format;uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;device->CreateUnorderedAccessView(image_.Get(),nullptr,&uav,heap_->GetCPUDescriptorHandleForHeapStart());
  D3D12_DESCRIPTOR_RANGE range={};range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV;range.NumDescriptors=1;range.OffsetInDescriptorsFromTableStart=0;
  D3D12_ROOT_PARAMETER params[3]={};params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[0].Descriptor.ShaderRegister=0;params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[1].DescriptorTable={1,&range};params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[2].Constants={0,0,2};
  D3D12_ROOT_SIGNATURE_DESC sig={};sig.NumParameters=3;sig.pParameters=params;ComPtr<ID3DBlob>blob,error;check(D3D12SerializeRootSignature(&sig,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"serialize convert root");check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_)),"convert root");
  const char*code=R"HLSL(ByteAddressBuffer src:register(t0);RWTexture2D<float4> dst:register(u0);cbuffer C:register(b0){uint width,height;}
float halfValue(uint i){uint bits=src.Load((i>>1)*4);return f16tof32((bits>>((i&1)*16))&65535);}
[numthreads(16,8,1)]void main(uint3 id:SV_DispatchThreadID){if(id.x>=width||id.y>=height)return;uint i=id.y*width+id.x,n=width*height;float3 rgb=float3(halfValue(i),halfValue(n+i),halfValue(2*n+i));rgb=floor(saturate(rgb)*255.0+0.5)/255.0;dst[id.xy]=float4(rgb,1);})HLSL";
  check(D3DCompile(code,strlen(code),"neo-dml-output",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error),"compile convert shader");D3D12_COMPUTE_PIPELINE_STATE_DESC ps={};ps.pRootSignature=root_.Get();ps.CS={blob->GetBufferPointer(),blob->GetBufferSize()};check(device->CreateComputePipelineState(&ps,IID_PPV_ARGS(&pipeline_)),"convert pipeline");
  event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event_)throw std::runtime_error("convert event");
 }
 void convert(ID3D12Resource*source,bool uav){
  auto desc=source->GetDesc();if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER||desc.Width<((UINT64(width_)*height_*6+3)&~UINT64(3)))throw std::runtime_error("unaligned or short DML output");
  check(allocator_->Reset(),"reset convert allocator");check(list_->Reset(allocator_.Get(),pipeline_.Get()),"reset convert list");auto state=uav?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_COMMON;
  D3D12_RESOURCE_BARRIER begin[2]={barrier(source,state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),barrier(image_.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};list_->ResourceBarrier(2,begin);
  ID3D12DescriptorHeap*heaps[]={heap_.Get()};list_->SetDescriptorHeaps(1,heaps);list_->SetComputeRootSignature(root_.Get());list_->SetComputeRootShaderResourceView(0,source->GetGPUVirtualAddress());list_->SetComputeRootDescriptorTable(1,heap_->GetGPUDescriptorHandleForHeapStart());UINT dims[]={width_,height_};list_->SetComputeRoot32BitConstants(2,2,dims,0);list_->Dispatch((width_+15)/16,(height_+7)/8,1);
  D3D12_RESOURCE_BARRIER end[2]={barrier(source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,state),barrier(image_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON)};list_->ResourceBarrier(2,end);check(list_->Close(),"close convert commands");inflight_source_=source;ID3D12CommandList*commands[]={list_.Get()};queue_->ExecuteCommandLists(1,commands);check(queue_->Signal(fence_.Get(),++value_),"signal convert fence");check(fence_->SetEventOnCompletion(value_,event_),"arm convert fence");if(WaitForSingleObject(event_,10000)!=WAIT_OBJECT_0)throw std::runtime_error("convert fence timeout");inflight_source_.Reset();context11_->CopyResource(display11_.Get(),image11_.Get());context11_->Flush();
 }
 // WGL writes display11_; D3D11 completion precedes D3D12 consumption.
 void pack(ID3D12Resource*destination){
  UINT64 bytes=(UINT64(width_)*height_*6+3)&~UINT64(3);
  if(destination->GetDesc().Width<bytes)throw std::runtime_error("short DML input");
  if(!pack_pipeline_){
   D3D12_SHADER_RESOURCE_VIEW_DESC srv={};srv.Format=DXGI_FORMAT_R8G8B8A8_UNORM;srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;srv.Texture2D.MipLevels=1;auto srv_handle=heap_->GetCPUDescriptorHandleForHeapStart();srv_handle.ptr+=device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);device_->CreateShaderResourceView(image_.Get(),&srv,srv_handle);
   D3D12_DESCRIPTOR_RANGE range={};range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV;range.NumDescriptors=1;
   D3D12_ROOT_PARAMETER params[3]={};params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[0].DescriptorTable={1,&range};params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[2].Constants={0,0,2};
   D3D12_ROOT_SIGNATURE_DESC sig={};sig.NumParameters=3;sig.pParameters=params;ComPtr<ID3DBlob>blob,error;check(D3D12SerializeRootSignature(&sig,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"pack root serialize");check(device_->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&pack_root_)),"pack root");
   const char*code=R"HLSL(Texture2D<float4> src:register(t0);RWByteAddressBuffer dst:register(u0);cbuffer C:register(b0){uint width,height;}
static const uint half_u8[256]={0,7172,8196,8710,9220,9477,9734,9991,10244,10373,10501,10630,10758,10887,11015,11144,11268,11332,11397,11461,11525,11589,11654,11718,11782,11846,11911,11975,12039,12103,12168,12232,12292,12324,12356,12388,12421,12453,12485,12517,12549,12581,12613,12645,12678,12710,12742,12774,12806,12838,12870,12902,12935,12967,12999,13031,13063,13095,13127,13159,13192,13224,13256,13288,13316,13332,13348,13364,13380,13396,13412,13428,13445,13461,13477,13493,13509,13525,13541,13557,13573,13589,13605,13621,13637,13653,13669,13685,13702,13718,13734,13750,13766,13782,13798,13814,13830,13846,13862,13878,13894,13910,13926,13942,13959,13975,13991,14007,14023,14039,14055,14071,14087,14103,14119,14135,14151,14167,14183,14199,14216,14232,14248,14264,14280,14296,14312,14328,14340,14348,14356,14364,14372,14380,14388,14396,14404,14412,14420,14428,14436,14444,14452,14460,14469,14477,14485,14493,14501,14509,14517,14525,14533,14541,14549,14557,14565,14573,14581,14589,14597,14605,14613,14621,14629,14637,14645,14653,14661,14669,14677,14685,14693,14701,14709,14717,14726,14734,14742,14750,14758,14766,14774,14782,14790,14798,14806,14814,14822,14830,14838,14846,14854,14862,14870,14878,14886,14894,14902,14910,14918,14926,14934,14942,14950,14958,14966,14974,14983,14991,14999,15007,15015,15023,15031,15039,15047,15055,15063,15071,15079,15087,15095,15103,15111,15119,15127,15135,15143,15151,15159,15167,15175,15183,15191,15199,15207,15215,15223,15231,15240,15248,15256,15264,15272,15280,15288,15296,15304,15312,15320,15328,15336,15344,15352,15360};
uint value(uint i){uint n=width*height;if(i>=3*n)return 0;uint c=i/n,p=i%n;float v=src.Load(int3(p%width,p/width,0))[c];return half_u8[(uint)floor(saturate(v)*255.0+0.5)];}
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID){uint i=2*(id.y*width+id.x);if(id.x>=width||i>=3*width*height)return;dst.Store(i*2,value(i)|(value(i+1)<<16));})HLSL";
   check(D3DCompile(code,strlen(code),"neo-dml-input",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error),"pack shader");D3D12_COMPUTE_PIPELINE_STATE_DESC ps={};ps.pRootSignature=pack_root_.Get();ps.CS={blob->GetBufferPointer(),blob->GetBufferSize()};check(device_->CreateComputePipelineState(&ps,IID_PPV_ARGS(&pack_pipeline_)),"pack pipeline");
   D3D11_QUERY_DESC query={D3D11_QUERY_EVENT,0};check(device11_->CreateQuery(&query,&pack_query_),"pack D3D11 event");
  }
  context11_->CopyResource(image11_.Get(),display11_.Get());context11_->End(pack_query_.Get());context11_->Flush();
  ULONGLONG deadline=GetTickCount64()+10000;BOOL done=FALSE;while(true){HRESULT hr=context11_->GetData(pack_query_.Get(),&done,sizeof(done),0);check(hr,"pack D3D11 sync");if(hr==S_OK&&done)break;if(GetTickCount64()>deadline)throw std::runtime_error("pack D3D11 timeout");SwitchToThread();}
  check(allocator_->Reset(),"reset pack allocator");check(list_->Reset(allocator_.Get(),pack_pipeline_.Get()),"reset pack list");auto begin=barrier(image_.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);list_->ResourceBarrier(1,&begin);
  ID3D12DescriptorHeap*heaps[]={heap_.Get()};list_->SetDescriptorHeaps(1,heaps);list_->SetComputeRootSignature(pack_root_.Get());auto srv_gpu=heap_->GetGPUDescriptorHandleForHeapStart();srv_gpu.ptr+=device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);list_->SetComputeRootDescriptorTable(0,srv_gpu);list_->SetComputeRootUnorderedAccessView(1,destination->GetGPUVirtualAddress());UINT dims[]={width_,height_};list_->SetComputeRoot32BitConstants(2,2,dims,0);list_->Dispatch((width_+15)/16,(((3*height_+1)/2)+15)/16,1);
  D3D12_RESOURCE_BARRIER end[2]={barrier(image_.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON),{}};end[1].Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;end[1].UAV.pResource=destination;list_->ResourceBarrier(2,end);check(list_->Close(),"close pack commands");inflight_source_=destination;ID3D12CommandList*commands[]={list_.Get()};queue_->ExecuteCommandLists(1,commands);check(queue_->Signal(fence_.Get(),++value_),"signal pack fence");check(fence_->SetEventOnCompletion(value_,event_),"arm pack fence");if(WaitForSingleObject(event_,10000)!=WAIT_OBJECT_0)throw std::runtime_error("pack fence timeout");inflight_source_.Reset();
 }
 void assemble(ID3D12Resource**sources,UINT count,ID3D12Resource*target,UINT pw,UINT ph,UINT time_bits){
  if((count!=2&&count!=4)||!pw||!ph||pw>8192||ph>8192)throw std::runtime_error("invalid interpolation geometry");UINT channels=count*3+(count==2?1:2);UINT64 bytes=(UINT64(pw)*ph*channels*2+3)&~UINT64(3);if(target->GetDesc().Width<bytes)throw std::runtime_error("short interpolation buffer");
  if(!interp_pipeline_){
   D3D12_ROOT_PARAMETER params[6]={};for(UINT i=0;i<4;++i){params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[i].Descriptor.ShaderRegister=i;}params[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;params[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[5].Constants={0,0,7};D3D12_ROOT_SIGNATURE_DESC sig={};sig.NumParameters=6;sig.pParameters=params;ComPtr<ID3DBlob>blob,error;check(D3D12SerializeRootSignature(&sig,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"interpolation root serialize");check(device_->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&interp_root_)),"interpolation root");
   const char*code=R"HLSL(ByteAddressBuffer a:register(t0);ByteAddressBuffer b:register(t1);ByteAddressBuffer c:register(t2);ByteAddressBuffer d:register(t3);RWByteAddressBuffer dst:register(u0);cbuffer C:register(b0){uint width,height,pw,ph,channels,frames,time_bits;}
uint halfBits(uint i){uint n=pw*ph,ch=i/n;if(ch>=channels)return 0;if(ch==3*frames)return time_bits;if(ch>3*frames)return 15360;uint p=i%n,idx=(ch%3)*width*height+min(p/pw,height-1)*width+min(p%pw,width-1);uint v=ch<3?a.Load((idx>>1)*4):ch<6?b.Load((idx>>1)*4):ch<9?c.Load((idx>>1)*4):d.Load((idx>>1)*4);return(v>>((idx&1)*16))&65535;}
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID){uint i=2*(id.y*pw+id.x);if(id.x>=pw||i>=channels*pw*ph)return;dst.Store(i*2,halfBits(i)|(halfBits(i+1)<<16));})HLSL";
   check(D3DCompile(code,strlen(code),"neo-dml-interpolation",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error),"interpolation shader");D3D12_COMPUTE_PIPELINE_STATE_DESC ps={};ps.pRootSignature=interp_root_.Get();ps.CS={blob->GetBufferPointer(),blob->GetBufferSize()};check(device_->CreateComputePipelineState(&ps,IID_PPV_ARGS(&interp_pipeline_)),"interpolation pipeline");
  }
  for(UINT i=0;i<count;++i)if(sources[i]->GetDesc().Width<((UINT64(width_)*height_*6+3)&~UINT64(3)))throw std::runtime_error("short interpolation history");
  check(allocator_->Reset(),"reset interpolation allocator");check(list_->Reset(allocator_.Get(),interp_pipeline_.Get()),"reset interpolation list");std::vector<D3D12_RESOURCE_BARRIER> barriers;for(UINT i=0;i<count;++i)barriers.push_back(barrier(sources[i],D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));list_->ResourceBarrier(count,barriers.data());list_->SetComputeRootSignature(interp_root_.Get());for(UINT i=0;i<4;++i)list_->SetComputeRootShaderResourceView(i,sources[i<count?i:0]->GetGPUVirtualAddress());list_->SetComputeRootUnorderedAccessView(4,target->GetGPUVirtualAddress());UINT dims[]={width_,height_,pw,ph,channels,count,time_bits};list_->SetComputeRoot32BitConstants(5,7,dims,0);list_->Dispatch((pw+15)/16,(((channels*ph+1)/2)+15)/16,1);
  barriers.clear();for(UINT i=0;i<count;++i)barriers.push_back(barrier(sources[i],D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS));D3D12_RESOURCE_BARRIER uav={};uav.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;uav.UAV.pResource=target;barriers.push_back(uav);list_->ResourceBarrier(UINT(barriers.size()),barriers.data());check(list_->Close(),"close interpolation commands");inflight_source_=target;inflight_inputs_.clear();for(UINT i=0;i<count;++i)inflight_inputs_.emplace_back(sources[i]);ID3D12CommandList*commands[]={list_.Get()};queue_->ExecuteCommandLists(1,commands);check(queue_->Signal(fence_.Get(),++value_),"signal interpolation fence");check(fence_->SetEventOnCompletion(value_,event_),"arm interpolation fence");if(WaitForSingleObject(event_,10000)!=WAIT_OBJECT_0)throw std::runtime_error("interpolation fence timeout");inflight_source_.Reset();inflight_inputs_.clear();
 }
 ID3D11Texture2D*texture(){return display11_.Get();}
 bool idle(){return !fence_||fence_->GetCompletedValue()>=value_||FAILED(device_->GetDeviceRemovedReason());}
 ~DmlTexture(){if(event_)CloseHandle(event_);}
};
// A timed-out submission must retain both its source and destination resources.
// Reclaim it when complete (or the device is removed); never free in-flight GPU memory.
static std::mutex retired_mutex;static std::vector<DmlTexture*>retired;
static void retire(DmlTexture*value)noexcept{try{std::lock_guard<std::mutex>guard(retired_mutex);for(size_t i=0;i<retired.size();){if(retired[i]->idle()){delete retired[i];retired.erase(retired.begin()+i);}else ++i;}if(value){if(value->idle())delete value;else retired.push_back(value);}}catch(...){/* retain ownership until process teardown if retirement bookkeeping fails */}}
extern "C" {
__declspec(dllexport) void*neo_dml_texture_create(void*device,unsigned w,unsigned h){try{bridge_error[0]=0;retire(nullptr);return new DmlTexture(static_cast<ID3D12Device*>(device),w,h);}catch(const std::exception&e){strncpy_s(bridge_error,e.what(),_TRUNCATE);return nullptr;}}
__declspec(dllexport) int neo_dml_texture_convert(void*handle,void*source,int uav){try{bridge_error[0]=0;static_cast<DmlTexture*>(handle)->convert(static_cast<ID3D12Resource*>(source),uav!=0);return 0;}catch(const std::exception&e){strncpy_s(bridge_error,e.what(),_TRUNCATE);return -1;}}
__declspec(dllexport) int neo_dml_texture_pack(void*handle,void*destination){try{bridge_error[0]=0;static_cast<DmlTexture*>(handle)->pack(static_cast<ID3D12Resource*>(destination));return 0;}catch(const std::exception&e){strncpy_s(bridge_error,e.what(),_TRUNCATE);return -1;}}
__declspec(dllexport) int neo_dml_interp_assemble(void*handle,void**sources,unsigned count,void*target,unsigned pw,unsigned ph,unsigned time_bits){try{bridge_error[0]=0;static_cast<DmlTexture*>(handle)->assemble(reinterpret_cast<ID3D12Resource**>(sources),count,static_cast<ID3D12Resource*>(target),pw,ph,time_bits);return 0;}catch(const std::exception&e){strncpy_s(bridge_error,e.what(),_TRUNCATE);return -1;}}
__declspec(dllexport) void*neo_dml_texture_get(void*handle){return static_cast<DmlTexture*>(handle)->texture();}
__declspec(dllexport) void neo_dml_texture_destroy(void*handle){retire(static_cast<DmlTexture*>(handle));}
__declspec(dllexport) const char*neo_dml_texture_error(){return bridge_error;}
}
