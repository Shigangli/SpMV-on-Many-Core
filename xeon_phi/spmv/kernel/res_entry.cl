#if defined BIT_8
#define BIT_TYPE char
#define BIT_WIDTH 8
#define BIT_WIDTH_1 7
#define I_MAX 255
#endif
#if defined BIT_16
#define BIT_TYPE short
#define BIT_WIDTH 16
#define BIT_WIDTH_1 15
#define I_MAX 65535
#endif
#if defined BIT_32
#define BIT_TYPE int
#define BIT_WIDTH 32
#define BIT_WIDTH_1 31
#define I_MAX UINT_MAX 
#endif
#if defined BIT_64
#define BIT_TYPE long 
#define BIT_WIDTH 64
#define BIT_WIDTH_1 63
#define I_MAX ULONG_MAX 
#endif

__kernel void res_entry(__global unsigned BIT_TYPE *bit,__global int *res_entry, 
#if defined COL_COM_1
                   __global short *col_delta,
                   __global unsigned int *col,
#endif
                   __global int * para_scan, int cta,
                   __global volatile int *inter,int groupnum)
{
    unsigned int lid = get_local_id(0);
    unsigned int tid = get_global_id(0);
    unsigned int gid = get_group_id(0);
    __local int column[2][NB_LSIZE],lsum;
    int src_id = tid*(cta/BIT_WIDTH);
    int sum=0,t1=0,t2=0,t3=0;
    if(tid!=0){
        for(int j=0;j<cta/BIT_WIDTH;j++){
            unsigned BIT_TYPE mask = 1<<BIT_WIDTH_1;
            unsigned BIT_TYPE bit_v = bit[src_id-cta/BIT_WIDTH+j];
            for(int k=0;k<BIT_WIDTH;k++,mask>>=1){
                sum += (bit_v&mask)>0?0:1;
            }
        }
    }
    else
        sum = 0;
    column[0][lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    //for(t1 =1,t3=1;t1<=NB_LSIZE/2; t1<<=1, t3 = t3^1){
        //column[t3][lid] = (lid >=t1) ? (column[t3^1][lid] + column[t3^1][lid - t2]) : column[t3^1][lid];
        //barrier(CLK_LOCAL_MEM_FENCE);

    for(t1 =1,t2 = 1,t3=1;t1<=NB_LSIZE/2; t1<<=1,t2 <<=1, t3 = t3^1){
        int puppet;
        if(lid>=t1)
            puppet=lid-t2;
        else
           puppet=0;
        column[t3][lid] = lid >=t1 ? column[t3^1][lid] + column[t3^1][puppet] : column[t3^1][lid];
        //column[t3][lid] = lid >=t1 ? column[t3^1][lid] + column[t3^1][lid - t2] : column[t3^1][lid];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum=0;
    t3=t3^1;
    if(lid==0)
    {
        if(gid == 0)
            inter[0] = column[t3][NB_LSIZE-1];
        else
        {
            while((sum=inter[gid - 1])==INT_MAX){}
            inter[gid] = column[t3][NB_LSIZE-1] + sum;
        }
        lsum = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_entry[tid] = lsum + column[t3][lid];
    if(tid==NB_LSIZE*groupnum-1){
        res_entry[tid+1]=res_entry[tid];
    }

    int seg=0;
    for(int j=0;j<cta/BIT_WIDTH;j++)
        if(bit[src_id+j]==I_MAX) seg=1;
    column[0][lid] = seg;
    for(int i=NB_LSIZE/2;i>0;i=i/2){
        if(lid<i){
            column[0][lid] += column[0][lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        if(column[0][0]==0)
            para_scan[gid]=0;
        else
            para_scan[gid]=1;
    }
#if defined COL_COM_1
    col_delta[tid] = -1;
    for(int j=1;j<cta;j++){
        int diff = col[tid + j*(groupnum * NB_LSIZE)] - col[tid + (j-1)*(groupnum * NB_LSIZE)];
        if( diff > 32767 || diff < -32767 ){
            col_delta[tid + j*(groupnum * NB_LSIZE)] = -1;
        }
        else{
            col_delta[tid + j*(groupnum * NB_LSIZE)] = (short)diff;
        }
    }
#endif
}
