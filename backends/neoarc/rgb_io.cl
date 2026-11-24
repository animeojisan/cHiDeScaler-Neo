// GPU color conversion / TRUE-N history. No inference or temporal synthesis.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void neoarc_rgb_input(__global const uchar *source,
    __global const half *previous, __global half *next,
    int pixels, int stride, int frames, int history_valid) {
    const int i=get_global_id(0);
    if(i>=pixels*3*frames) return;
    const int channel=i/pixels, pixel=i%pixels, frame=channel/3;
    if(history_valid && frame+1<frames) next[i]=previous[i+3*pixels];
    else next[i]=convert_half_rte(convert_float(source[pixel*stride+channel%3])*(1.0f/255.0f));
}
__kernel void neoarc_rgb_f16_output(__global const half *source,
    __global uchar *output, int pixels, int stride) {
    const int i=get_global_id(0);
    if(i>=pixels) return;
    for(int channel=0;channel<3;++channel) {
        const float x=clamp(convert_float(source[channel*pixels+i]),0.0f,1.0f);
        output[i*stride+channel]=convert_uchar_sat_rtz(x*255.0f+0.5f);
    }
    if(stride==4) output[i*stride+3]=(uchar)255;
}
__kernel void neoarc_rgb_f32_output(__global const float *source,
    __global uchar *output, int pixels, int stride) {
    const int i=get_global_id(0);
    if(i>=pixels) return;
    for(int channel=0;channel<3;++channel) {
        const float x=clamp(source[channel*pixels+i],0.0f,1.0f);
        output[i*stride+channel]=convert_uchar_sat_rtz(x*255.0f+0.5f);
    }
    if(stride==4) output[i*stride+3]=(uchar)255;
}

const sampler_t neoarc_gl_sampler=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;
__kernel void neoarc_gl_input(read_only image2d_t source,
    __global const half *previous,__global half *next,
    int width,int height,int frames,int history_valid) {
    const int pixels=width*height,i=get_global_id(0);
    if(i>=pixels*3*frames)return;
    const int channel=i/pixels,p=i%pixels;
    if(history_valid&&channel/3+1<frames)next[i]=previous[i+3*pixels];
    else {
        const float4 rgba=read_imagef(source,neoarc_gl_sampler,(int2)(p%width,p/width));
        // Preserve the existing GL->RGB8->FP16 boundary, including pre-GLSL.
        const float value=floor(clamp(rgba[channel%3],0.0f,1.0f)*255.0f+0.5f)*(1.0f/255.0f);
        next[i]=convert_half_rte(value);
    }
}
__kernel void neoarc_gl_f16_output(__global const half *source,write_only image2d_t output,
    int width,int height) {
    const int i=get_global_id(0),pixels=width*height;if(i>=pixels)return;
    float4 value=(float4)(convert_float(source[i]),convert_float(source[pixels+i]),convert_float(source[2*pixels+i]),1.0f);
    value=floor(clamp(value,0.0f,1.0f)*255.0f+0.5f)*(1.0f/255.0f);value.w=1.0f;
    write_imagef(output,(int2)(i%width,i/width),value);
}
__kernel void neoarc_gl_f32_output(__global const float *source,write_only image2d_t output,
    int width,int height) {
    const int i=get_global_id(0),pixels=width*height;if(i>=pixels)return;
    float4 value=(float4)(source[i],source[pixels+i],source[2*pixels+i],1.0f);
    value=floor(clamp(value,0.0f,1.0f)*255.0f+0.5f)*(1.0f/255.0f);value.w=1.0f;
    write_imagef(output,(int2)(i%width,i/width),value);
}

// Native RIFE geometry. Always upload both current endpoints; no pointer cache.
__kernel void neoarc_rife_rgb_input(__global const uchar *source,__global half *output,
    int pixels,int stride,float phase) {
    const int i=get_global_id(0);if(i>=7*pixels)return;
    const int channel=i/pixels,p=i%pixels;
    if(channel==6)output[i]=convert_half_rte(clamp(phase,0.0f,1.0f));
    else output[i]=convert_half_rte(convert_float(source[(channel/3)*pixels*stride+p*stride+channel%3])*(1.0f/255.0f));
}

// Explicit ordered frame textures. No history identity guessing or host pixels.
__kernel void neoarc_gl_interp_input(read_only image2d_t a,read_only image2d_t b,
    read_only image2d_t c,read_only image2d_t d,__global half *out,
    int width,int height,int channels,int frames,float phase) {
    const int n=width*height,i=get_global_id(0);if(i>=n*channels)return;
    const int ch=i/n,p=i%n;const int2 xy=(int2)(p%width,p/width);
    float value;
    if(ch>=3*frames) value=ch==3*frames?clamp(phase,0.0f,1.0f):1.0f;
    else {
        float4 v;switch(ch/3){case 0:v=read_imagef(a,neoarc_gl_sampler,xy);break;
        case 1:v=read_imagef(b,neoarc_gl_sampler,xy);break;
        case 2:v=read_imagef(c,neoarc_gl_sampler,xy);break;
        default:v=read_imagef(d,neoarc_gl_sampler,xy);break;}
        value=floor(clamp(v[ch%3],0.0f,1.0f)*255.0f+0.5f)*(1.0f/255.0f);
    }
    out[i]=convert_half_rte(value);
}

inline half3 neoarc_image_half3(float4 v) {
    const float3 q=floor(clamp(v.xyz,0.0f,1.0f)*255.0f+0.5f)*(1.0f/255.0f);
    return convert_half3_rte(q);
}
__kernel void neoarc_gl_input_vector(read_only image2d_t source,
    __global const half *previous,__global half *next,
    int width,int height,int frames,int history_valid) {
    const int n=width*height,p=get_global_id(0);if(p>=n)return;
    const half3 current=neoarc_image_half3(read_imagef(source,neoarc_gl_sampler,(int2)(p%width,p/width)));
    for(int f=0;f<frames;++f){
        const int ch=3*f;
        if(history_valid&&f+1<frames){next[ch*n+p]=previous[(ch+3)*n+p];next[(ch+1)*n+p]=previous[(ch+4)*n+p];next[(ch+2)*n+p]=previous[(ch+5)*n+p];}
        else {next[ch*n+p]=current.x;next[(ch+1)*n+p]=current.y;next[(ch+2)*n+p]=current.z;}
    }
}
__kernel void neoarc_gl_interp_input_vector(read_only image2d_t a,read_only image2d_t b,
    read_only image2d_t c,read_only image2d_t d,__global half *out,
    int width,int height,int channels,int frames,float phase) {
    const int n=width*height,p=get_global_id(0);if(p>=n)return;const int2 xy=(int2)(p%width,p/width);
    const half3 va=neoarc_image_half3(read_imagef(a,neoarc_gl_sampler,xy));
    const half3 vb=neoarc_image_half3(read_imagef(b,neoarc_gl_sampler,xy));
    out[p]=va.x;out[n+p]=va.y;out[2*n+p]=va.z;out[3*n+p]=vb.x;out[4*n+p]=vb.y;out[5*n+p]=vb.z;
    if(frames==4){const half3 vc=neoarc_image_half3(read_imagef(c,neoarc_gl_sampler,xy));const half3 vd=neoarc_image_half3(read_imagef(d,neoarc_gl_sampler,xy));
        out[6*n+p]=vc.x;out[7*n+p]=vc.y;out[8*n+p]=vc.z;out[9*n+p]=vd.x;out[10*n+p]=vd.y;out[11*n+p]=vd.z;out[13*n+p]=(half)1.0f;}
    out[3*frames*n+p]=convert_half_rte(clamp(phase,0.0f,1.0f));
}
// v985 buffer input: the two RIFE endpoints as RGBA FP16 pixel buffers (GL
// pixel-pack copies of the frame textures, shared through clCreateFromGLBuffer).
// Same quantization as the image path (neoarc_image_half3), so the packed
// tensor is identical; it only avoids the per-enqueue cost of GL images.
__kernel void neoarc_gl_interp_input_buffer(__global const half *a,__global const half *b,
    __global half *out,int width,int height,float phase) {
    const int n=width*height,p=get_global_id(0);if(p>=n)return;
    const half3 va=neoarc_image_half3(vload_half4(p,a));
    const half3 vb=neoarc_image_half3(vload_half4(p,b));
    out[p]=va.x;out[n+p]=va.y;out[2*n+p]=va.z;out[3*n+p]=vb.x;out[4*n+p]=vb.y;out[5*n+p]=vb.z;
    out[6*n+p]=convert_half_rte(clamp(phase,0.0f,1.0f));
}
// v985b staged input: both endpoints in one CL-owned buffer (frame 0 RGBA FP16
// pixels, then frame 1). Filled on the GL owner thread, so the inference worker
// never acquires GL buffers. Same quantization as the image path.
__kernel void neoarc_gl_interp_input_staged(__global const half *ab,
    __global half *out,int width,int height,float phase) {
    const int n=width*height,p=get_global_id(0);if(p>=n)return;
    const half3 va=neoarc_image_half3(vload_half4(p,ab));
    const half3 vb=neoarc_image_half3(vload_half4(n+p,ab));
    out[p]=va.x;out[n+p]=va.y;out[2*n+p]=va.z;out[3*n+p]=vb.x;out[4*n+p]=vb.y;out[5*n+p]=vb.z;
    out[6*n+p]=convert_half_rte(clamp(phase,0.0f,1.0f));
}
__kernel void neoarc_gl_interp_time(__global half *input,int pixels,int channel,float phase) {
    const int p=get_global_id(0);if(p<pixels)input[channel*pixels+p]=convert_half_rte(clamp(phase,0.0f,1.0f));
}

// v983 boost scene-cut gate. Same thumbnail as NeoAMD v133 boost_scene_cut:
// the two packed RIFE endpoints (planes 0..5, FP16 of 8-bit/255) resized to
// 64x36 with half_pixel bilinear sampling, each sample rounded to FP16.
__kernel void neoarc_scene_thumb(__global const half *input,__global half *thumb,int width,int height) {
    const int i=get_global_id(0);if(i>=6*36*64)return;
    const int plane=i/(36*64),r=i%(36*64),y=r/64,x=r%64,n=width*height;
    float sy=((float)y+0.5f)*(float)height/36.0f-0.5f,sx=((float)x+0.5f)*(float)width/64.0f-0.5f;
    sx=clamp(sx,0.0f,(float)(width-1));sy=clamp(sy,0.0f,(float)(height-1));
    const int x0=(int)floor(sx),y0=(int)floor(sy),x1=min(x0+1,width-1),y1=min(y0+1,height-1);
    const float ax=sx-(float)x0,ay=sy-(float)y0;
    __global const half *p=input+plane*n;
    const float v00=convert_float(p[y0*width+x0]),v10=convert_float(p[y0*width+x1]);
    const float v01=convert_float(p[y1*width+x0]),v11=convert_float(p[y1*width+x1]);
    const float t0=fma(ax,v10-v00,v00),t1=fma(ax,v11-v01,v01);
    thumb[i]=convert_half_rte(fma(ay,t1-t0,t0));
}
// v983 boost scene-cut lane: copy one packed endpoint into the model output
// tensor (frame 0 for t<0.5, frame 1 otherwise). The regular GL output kernel
// then quantizes it back to the exact 8-bit source picture.
__kernel void neoarc_cut_copy_f16(__global const half *input,__global half *output,int pixels,int frame) {
    const int i=get_global_id(0);if(i<3*pixels)output[i]=input[frame*3*pixels+i];
}
__kernel void neoarc_cut_copy_f32(__global const half *input,__global float *output,int pixels,int frame) {
    const int i=get_global_id(0);if(i<3*pixels)output[i]=convert_float(input[frame*3*pixels+i]);
}

// v983 native boost head for the Lite RIFE final stage ("diagparallel"), the
// same math as the ONNX head in FP32: flow0*2t, flow1*2(1-t), mask
// m(1-t)/(m(1-t)+(1-m)t), normalized base grid x/(W-1)*2-1, one bilinear
// GridSample (align_corners=1, border) of frames 0/1 and the mask blend.
// state = [flow0 x,y, flow1 x,y, mask] planes at the input geometry.
inline float neoarc_bh_sample(__global const half *plane,int width,int height,float gx,float gy) {
    float fx=(gx+1.0f)*0.5f*(float)(width-1),fy=(gy+1.0f)*0.5f*(float)(height-1);
    // fmax/fmin return the non-NaN operand: a NaN/Inf flow can never index
    // outside the plane (the ONNX GridSample is equally bounded).
    fx=fmin(fmax(fx,0.0f),(float)(width-1));fy=fmin(fmax(fy,0.0f),(float)(height-1));
    const int x0=clamp((int)floor(fx),0,width-1),y0=clamp((int)floor(fy),0,height-1);
    const int x1=min(x0+1,width-1),y1=min(y0+1,height-1);
    const float ax=fx-(float)x0,ay=fy-(float)y0;
    const float v00=convert_float(plane[y0*width+x0]),v10=convert_float(plane[y0*width+x1]);
    const float v01=convert_float(plane[y1*width+x0]),v11=convert_float(plane[y1*width+x1]);
    return (v00*(1.0f-ax)+v10*ax)*(1.0f-ay)+(v01*(1.0f-ax)+v11*ax)*ay;
}
#define NEOARC_BOOST_HEAD(NAME,ST,OT,STORE) \
__kernel void NAME(__global const half *input,__global const ST *state,__global OT *output,int width,int height,float t) { \
    const int n=width*height,p=get_global_id(0);if(p>=n)return; \
    const int x=p%width,y=p/width;const float r=1.0f-t,a=2.0f*t,b=2.0f*r; \
    const float bx=(float)x/(float)(width-1)*2.0f-1.0f,by=(float)y/(float)(height-1)*2.0f-1.0f; \
    const float g0x=bx+convert_float(state[p])*a,g0y=by+convert_float(state[n+p])*a; \
    const float g1x=bx+convert_float(state[2*n+p])*b,g1y=by+convert_float(state[3*n+p])*b; \
    const float m=convert_float(state[4*n+p]),w0=m*r,w1=(1.0f-m)*t,kr=w0/(w0+w1),k=isfinite(kr)?kr:0.5f; \
    for(int c=0;c<3;++c){ \
        const float v0=neoarc_bh_sample(input+c*n,width,height,g0x,g0y); \
        const float v1=neoarc_bh_sample(input+(3+c)*n,width,height,g1x,g1y); \
        output[c*n+p]=STORE(v0*k+v1*(1.0f-k)); \
    } \
}
#define NEOARC_STORE_HALF(v) convert_half_rte(v)
#define NEOARC_STORE_FLOAT(v) (v)
NEOARC_BOOST_HEAD(neoarc_boost_head_s16_o16,half,half,NEOARC_STORE_HALF)
NEOARC_BOOST_HEAD(neoarc_boost_head_s16_o32,half,float,NEOARC_STORE_FLOAT)
NEOARC_BOOST_HEAD(neoarc_boost_head_s32_o16,float,half,NEOARC_STORE_HALF)
NEOARC_BOOST_HEAD(neoarc_boost_head_s32_o32,float,float,NEOARC_STORE_FLOAT)
