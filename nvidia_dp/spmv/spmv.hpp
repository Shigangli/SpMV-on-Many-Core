#include "../init.hpp"
#include "class.hpp"
#include "mtx.hpp"
#include "bccoo.hpp"
#include "cpu_spmv.hpp"
extern TimeRcd timeRcd;

template<class dataType>
void footPrintSort(int block_size[12][3],MTX<dataType> *mtx)
{
    int k=0;
    for(int width=1;width<=4;width=width*2){
        for(int height=1;height<=4;height++){
            int block_number = getBlockNumber(mtx->row,mtx->col,mtx->cols,mtx->nnz,width,height);
            int bandwidth=0;
            if((mtx->cols+width-1)/width < 65535)
                bandwidth = block_number*height*width*sizeof(dataType)+block_number*sizeof(short)+block_number/8;
            else
                bandwidth = block_number*height*width*sizeof(dataType)+block_number*sizeof(int)+block_number/8;
             
            int k=-1;
            for(int bs=11;bs>=0;bs--){
                if(block_size[bs][2]>bandwidth)
                    k=bs;
            }
            if(k!=-1){
                for(int bs=11;bs>k;bs--){
                    block_size[bs][2]=block_size[bs-1][2];
                    block_size[bs][1]=block_size[bs-1][1];
                    block_size[bs][0]=block_size[bs-1][0];
                }
                block_size[k][2]=bandwidth;
                block_size[k][1]=width;
                block_size[k][0]=height;
            }
        }
    }
}

template<class dataType>
void generateProgramCache(clContext *clCxt)
{
    Plan *plan = (Plan *)malloc(sizeof(Plan));
    cl_ulong cstart,cend;
    struct timeval vstart,vend;
    FILE *fp = fopen("clbin/sign", "rb");
    if(fp != NULL){
        fclose(fp);
        return;
    }
    gettimeofday(&vstart,NULL);
#if defined PERF
    cout<<"Generate PTX File & Program Cache ... "<<endl;
#endif
#if defined ESTIMATE
    for(int trans=1;trans<=1;trans++){
    for(int tx=0;tx<=0;tx++){
    for(int coalesced=0;coalesced<=2;coalesced++){
    for(int logp=0;logp<=0;logp++){
    for(int col_delta=0;col_delta<=1;col_delta++){
#else
    for(int trans=0;trans<=1;trans++){
    for(int tx=0;tx<=0;tx++){
    for(int coalesced=0;coalesced<=4;coalesced++){
    for(int logp=0;logp<=1;logp++){
    for(int col_delta=0;col_delta<=1;col_delta++){
#endif
    for(int lt=64;lt<=MAX_WORKGROUP_SIZE;lt<<=1){
    for(int regp=0;regp<=4;regp++){
    for(int bitwidth=8;bitwidth<=32;bitwidth=bitwidth*2){
    for(int width=1;width<=4;width=width*2){
    for(int height=1;height<=4;height++){
    for(int dimwidth=16;dimwidth<=32;dimwidth=dimwidth*2){
        if(regp*bitwidth>128) continue;
        if(trans==0&&regp>1) continue;
        if(width==0||height==0) continue;
        if(col_delta==1&&(trans==0||dimwidth==16||width+height>3)) continue;
        if(trans==0&&coalesced!=0) continue;
        if(coalesced==0&&regp>1) continue;
        if(logp==0&&trans==0) continue;
        if(coalesced==0&&(lt*(height==3?4:height)*bitwidth>SH_MEM_SIZE/4))  continue;
        if(trans==1&&coalesced!=0&&(regp==0||logp!=0)) continue;
        if(regp==0&&logp==0) continue;
        plan->tx = 0;
        plan->trans = trans;
        plan->coalesced = coalesced;
        plan->localthread = lt; 
        plan->registergroup = regp;
        plan->localmemgroup = logp;
        plan->cta = bitwidth*(plan->registergroup + plan->localmemgroup);
        plan->block_width = width;
        plan->block_height = height;
        plan->col_delta = col_delta;
        plan->bitwidth = bitwidth;
        plan->dimwidth = dimwidth;
        char build_options[200];
        int registersize = plan->bitwidth*plan->registergroup;
        int localmemsize = plan->bitwidth*plan->localmemgroup;
        if(plan->coalesced==0){
            sprintf(build_options ,
            "-D DIM_%d -D TRANS_%d -D TX_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_SIZE=%d -D NB_LOCAL_SIZE=%d -D NB_CTA_%d -D BIT_%d -D SEGSCAN_%d -D COL_COM_%d",
                plan->dimwidth/8,plan->trans,plan->tx,plan->block_width,plan->block_height,
                TEXTURE_WIDTH,plan->localthread,registersize,localmemsize,localmemsize,
                plan->bitwidth,1,plan->col_delta);
            getProgram("spmv_strategy_a.cl",build_options,clCxt);
            cout<<build_options<<endl;
        }
        else if(plan->trans==1){
            sprintf(build_options ,
            "-D DIM_%d -D TX_%d -D TRANS_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_GRP=%d -D BIT_%d -D SEGSCAN_%d -D CACHE_LEN=%d -D COL_COM_%d",
                plan->dimwidth/8,plan->tx,plan->trans,plan->block_width,plan->block_height,
                TEXTURE_WIDTH,plan->localthread,plan->registergroup,plan->bitwidth,
                1,plan->coalesced*plan->localthread,plan->col_delta);
            getProgram("spmv_strategy_b.cl",build_options,clCxt);
            cout<<build_options<<endl;
        }
    }}}}}}}}}}}
    fp = fopen("clbin/sign", "wb+");
    if(fp != NULL)
    {
        fwrite("OK", 2, 1, fp);
        fclose(fp);
    }
    gettimeofday(&vend,NULL);
    cstart=(cl_ulong)vstart.tv_sec*1000000 + (cl_ulong)vstart.tv_usec;
    cend=(cl_ulong)vend.tv_sec*1000000 + (cl_ulong)vend.tv_usec;
#if defined PERF
    cout<<"Cost Time:"<<cend - cstart<<endl;
#endif
}

template<class dataType>
void getTexture(clContext *clCxt,cl_mem &vec_dev, dataType *vec, int vec_size,
                int block_width,int tx_width=TEXTURE_WIDTH){
    int tx_height=0;
    cl_channel_order channel_order; 
    if(block_width == 1)
        channel_order = CL_R;
    if(block_width == 2)
        channel_order = CL_RG;
    if(block_width == 3)
        channel_order = CL_RGB;
    if(block_width == 4)
        channel_order = CL_RGBA;
    const cl_image_format floatFormat=
    {
        channel_order,
        CL_FLOAT,
    };
    tx_height = (vec_size/sizeof(dataType)  + tx_width - 1)/tx_width;
    if (tx_height % block_width != 0)
    tx_height += (block_width - (tx_height % block_width));
    dataType* tex2dVec = (dataType*)malloc(sizeof(dataType)*tx_width*tx_height);
    memset(tex2dVec, 0, sizeof(dataType)*tx_width*tx_height);
    for (int i = 0; i < vec_size/sizeof(dataType); i++)
    {
        tex2dVec[i] = vec[i];
    }
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {tx_width, tx_height/block_width, 1};
    int errorCode = 0;
    vec_dev = clCreateImage2D(clCxt->context, CL_MEM_READ_ONLY, &floatFormat, 
                                           tx_width, tx_height/block_width, 0, NULL, &errorCode); 
    if(errorCode!=CL_SUCCESS) cout<<"Error!"<<__LINE__<<endl;
    errorCode = clEnqueueWriteImage(clCxt->command_queue, vec_dev, CL_TRUE, origin, 
                                                vectorSize, 0, 0, tex2dVec, 0, NULL, NULL);
    clFinish(clCxt->command_queue);
    if(errorCode!=CL_SUCCESS) cout<<"Error!"<<__LINE__<<endl;
    free(tex2dVec);
}

void transpose(clContext *clCxt,cl_mem &gdst, cl_mem &gsrc,int src_data_size,
               int block_width,int cta,int workgroup, int localthread,int trans){
    vector<pair<size_t ,const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&gdst ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&gsrc ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&cta ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&src_data_size ));
    size_t globalthreads[3] = {localthread * workgroup,1,1};
    size_t localthreads[3] = {localthread,1,1};
    cl_kernel kernel[1];
    vector<pair<size_t * ,size_t *> > threads;
    threads.push_back( make_pair( globalthreads, localthreads) );
    int trans_size = 32;
    if(cta % 32 !=0 ){
        trans_size = 16;
    }
    if(cta % 16 != 0){
        trans_size = 8;
    }
    if(block_width<=4){ 
        while(trans_size * block_width * localthread / 1024 > 8){
            trans_size = trans_size / 2;
        }
    }
    else{
        while(block_width==32&trans_size * localthread / 1024 > 8){
            trans_size = trans_size / 2;
        }
        while(block_width==16&trans_size * localthread / 1024 > 16){
            trans_size = trans_size / 2;
        }
    }
    char build_options[200];
    if(trans == 1)
        sprintf(build_options ,"-D KERNEL_NAME=TRANSPOSE -D NB_VEC_%d -D TRANS_SIZE=%d -D NB_LSIZE=%d",
               block_width,trans_size,localthread);
    else
        sprintf(build_options ,"-D KERNEL_NAME=TRANSPOSE -D NB_VEC_%d -D NB_LSIZE=%d",
               block_width,localthread);
    //cout<<build_options<<endl;
    kernel[0] = getKernel("transpose.cl","transpose",args,build_options,clCxt);
    executeKernel(threads,kernel,clCxt,1,0);
}

void getResEntryGpu(clContext *clCxt,cl_mem &gbit,cl_mem &res_dev_entry,
                       cl_mem &gcol,cl_mem &gcol_delta,cl_mem &gpara_scan,
                       int workgroup, int localthread,int cta,int bitwidth,int col_delta){
    int *inter = new int[workgroup];
    cl_mem ginter;
    for(int i=0;i<workgroup;i++) inter[i]=CL_INT_MAX;
    create(clCxt,&ginter,(workgroup)*sizeof(int)); 
    upload(clCxt,(void *)inter,ginter,(workgroup)*sizeof(int));
    vector<pair<size_t ,const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&gbit ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&res_dev_entry ));
    if(col_delta==1){
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&gcol_delta ));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&gcol ));
    }
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&gpara_scan ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&cta ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&ginter ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&workgroup ));
    char build_options[200];
    sprintf(build_options ,"-D KERNEL_NAME=RES_ENTRY -D BIT_%d -D COL_COM_%d -D NB_LSIZE=%d",
        bitwidth,col_delta,localthread);
    size_t globalthreads[3] = {localthread * workgroup,1,1};
    size_t localthreads[3] = {localthread,1,1};
    cl_kernel kernel[1];
    vector<pair<size_t * ,size_t *> > threads;
    threads.push_back( make_pair( globalthreads, localthreads) );
    kernel[0] = getKernel("res_entry.cl","res_entry",args,build_options,clCxt);
    executeKernel(threads,kernel,clCxt,1,0);
    delete[] inter;
    clReleaseMemObject(ginter);
}

template<class dataType>
void yaSpMVRun(clContext *clCxt,CLBCCOO *clbccoo,cl_mem vec_dev,cl_mem res_dev,Plan *plan,int times=1,int record=0)
{
    vector<pair<size_t ,const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->data) ));
    if(plan->block_height>1)
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->data1) ));
    if(plan->block_height>2)
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->data2) ));
    if(plan->block_height>3)
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->data3) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->col) ));
    if(plan->col_delta==1)
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->col_delta) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->bit) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->res_entry) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->inter) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->para_scan) ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&plan->workgroup ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&vec_dev ));
    cl_mem inter_res_dev;
    int inter_res_size = clbccoo->slice_rows*sizeof(dataType);
    if(clbccoo->slices!=1) {
        create(clCxt,&inter_res_dev,inter_res_size); 
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&inter_res_dev ));
    }
    else
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&res_dev ));
    int registersize = plan->bitwidth*plan->registergroup;
    int localmemsize = plan->bitwidth*plan->localmemgroup;
    char build_options[200];
    if(plan->coalesced==0){
        sprintf(build_options ,
            "-D DIM_%d -D TRANS_%d -D TX_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_SIZE=%d -D NB_LOCAL_SIZE=%d -D NB_CTA_%d -D BIT_%d -D SEGSCAN_1 -D COL_COM_%d",
            plan->dimwidth/8,plan->trans,plan->tx,plan->block_width,plan->block_height,
            TEXTURE_WIDTH,plan->localthread,registersize,localmemsize,localmemsize,plan->bitwidth,
            plan->col_delta);
    }
    else if(plan->trans==1){
        sprintf(build_options ,
            "-D DIM_%d -D TX_%d -D TRANS_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_GRP=%d -D BIT_%d -D SEGSCAN_1 -D CACHE_LEN=%d -D COL_COM_%d",
            plan->dimwidth/8,plan->tx,plan->trans,plan->block_width,plan->block_height,
            TEXTURE_WIDTH,plan->localthread,plan->registergroup,plan->bitwidth,
            plan->coalesced*plan->localthread,plan->col_delta);
    }
    else
        cout<<"The parameter of plan is unreasonable!"<<endl;
#if defined DEBUG
    //cout<<build_options<<endl;
#endif
    size_t globalthreads[3] = {plan->localthread * plan->workgroup,1,1};
    size_t localthreads[3] = {plan->localthread,1,1};
    cl_kernel kernel[2];
    vector<pair<size_t * ,size_t *> > threads;
    threads.push_back( make_pair( globalthreads, localthreads) );
    if(plan->coalesced == 0)
        kernel[0] = getKernel("spmv_strategy_a.cl","spmv_bccoo",args,build_options,clCxt);
    else
        kernel[0] = getKernel("spmv_strategy_b.cl","spmv_bccoo",args,build_options,clCxt);
    args.clear();
    if(clbccoo->slices!=1){
        int t_row=clbccoo->slice_rows/clbccoo->slices;
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&inter_res_dev ));
        args.push_back( make_pair( sizeof(cl_mem) , (void *)&res_dev ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&t_row ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&clbccoo->rows ));
        args.push_back( make_pair( sizeof(cl_int) , (void *)&clbccoo->slices ));
        size_t globalthreads_2[3] = {(clbccoo->slice_rows+255)/256*256,1,1};
        size_t localthreads_2[3] = {256,1,1};
        threads.push_back( make_pair( globalthreads_2, localthreads_2) );
        kernel[1] = getKernel("res_accumulate.cl","res_accumulate",args,"-D KERNELNAME=RES_ACCUMULATE",clCxt);
    }
    timeRcd.totaltime = 0;
    executeKernel(threads,kernel,clCxt,times,record);
    if(clbccoo->slices!=1)
        clReleaseMemObject(inter_res_dev);
    for(int i=0;i<threads.size();i++){
        cl_int ret = clReleaseKernel(kernel[i]);
        if(ret != CL_SUCCESS) cout << "Failed to release kernel. error code:"<< ret  << endl;
    }
}

template <class dataType,class dimType,class bitType>
#if defined DEBUG
void getPlan(clContext *clCxt,BCCOO<dataType,dimType,bitType> *bccoo,Plan *best,dataType *cres)
#else
void getPlan(clContext *clCxt,BCCOO<dataType,dimType,bitType> *bccoo,Plan *best)
#endif
{
    Plan *plan=(Plan *)malloc(sizeof(Plan));
    CLBCCOO *clbccoo = (CLBCCOO *)malloc(sizeof(CLBCCOO));
    cl_mem src_data,src_data1,src_data2,src_data3,src_col,vec_dev_tx,vec_dev;
    int vec_size = bccoo->cols * sizeof(dataType);
    int res_size = bccoo->rows * sizeof(dataType);
    dataType *vec = new dataType[vec_size/sizeof(dataType)];
    for(int i=0;i<vec_size/sizeof(dataType);i++) vec[i]=i;
    create(clCxt,&vec_dev,vec_size);
    upload(clCxt,(void *)vec,vec_dev,vec_size);
    getTexture(clCxt,vec_dev_tx,vec,vec_size,bccoo->block_width,TEXTURE_WIDTH);
    delete[] vec;
    int src_bit_size = bccoo->block_number/8;
    int src_data_size = bccoo->block_width * bccoo->block_number*sizeof(dataType);
    int src_col_size = bccoo->block_number*sizeof(dimType);
    create(clCxt,&src_col,src_col_size); 
    upload(clCxt,(void *)bccoo->col,src_col,src_col_size);
    create(clCxt,&src_data,src_data_size); 
    if(bccoo->block_height>1)  create(clCxt,&src_data1,src_data_size); 
    if(bccoo->block_height>2)  create(clCxt,&src_data2,src_data_size); 
    if(bccoo->block_height>3)  create(clCxt,&src_data3,src_data_size); 
    upload(clCxt,(void *)bccoo->data,src_data,src_data_size);
    if(bccoo->block_height>1)
        upload(clCxt,(void *)(&(bccoo->data[src_data_size/sizeof(dataType)])),src_data1,src_data_size);
    if(bccoo->block_height>2)
        upload(clCxt,(void *)(&(bccoo->data[2*src_data_size/sizeof(dataType)])),src_data2,src_data_size);
    if(bccoo->block_height>3)
        upload(clCxt,(void *)(&(bccoo->data[3*src_data_size/sizeof(dataType)])),src_data3,src_data_size);
    plan->dimwidth=sizeof(dimType)*8;
    plan->block_width = bccoo->block_width;
    plan->block_height = bccoo->block_height;
    plan->slices = bccoo->slices;
    clbccoo->rows = bccoo->rows;
    clbccoo->cols = bccoo->cols;
    clbccoo->nnz = bccoo->nnz;
    clbccoo->block_width = bccoo->block_width;
    clbccoo->block_height = bccoo->block_height;
    clbccoo->max_block_per_row = bccoo->max_block_per_row;
    clbccoo->slice_rows = bccoo->slice_rows;
    clbccoo->slices = bccoo->slices;
    for(int lt=64;lt<=MAX_WORKGROUP_SIZE;lt<<=1){
    for(int bw=8;bw<=32;bw<<=1){
#if defined ESTIMATE
    for(int gp=1;gp<=4;gp++){
    for(int tr=1;tr<=1;tr++){
    for(int cd=0;cd<=1;cd++){
#else
    for(int gp=1;gp<=5;gp++){
    for(int tr=0;tr<=1;tr++){
    for(int cd=0;cd<=1;cd++){
#endif
        if(cd==1&&(tr==0||sizeof(dimType)==2||plan->block_width+plan->block_height>3)) continue;
        if(tr==0&&gp>2) continue;
#if defined PERF
        cout<<"dim:"<<plan->dimwidth<<" bw:"<<bw<<" wid:"<<plan->block_width<<" hei:"<<plan->block_height
            <<" lt:"<<lt<<" sli:"<<plan->slices<<" tr:"<<tr <<" cd:"<<cd<<" cta:"<<gp*bw<<endl;
#endif
        plan->localthread = lt; 
        plan->bitwidth=bw;
        plan->trans = tr;
        plan->cta = gp*bw;
        plan->col_delta = cd;
        clbccoo->block_number = bccoo->block_number - bccoo->block_number % (lt * gp*bw) + (lt * gp*bw);
        plan->workgroup = clbccoo->block_number/(plan->localthread*plan->cta);

        int para_scan_size = plan->workgroup*sizeof(int);
        int inter_size = plan->workgroup*sizeof(dataType)*(plan->block_height>2?4:plan->block_height);
        int data_size = plan->block_width * clbccoo->block_number*sizeof(dataType);
        int col_size = clbccoo->block_number*sizeof(dimType);
        int bit_size = clbccoo->block_number/8;
        int res_entry_size = (plan->workgroup * plan->localthread+1) * sizeof(int);
        int col_delta_size = clbccoo->block_number * sizeof(short);
        unsigned char *new_bit = (unsigned char *)malloc(bit_size);
        new_bit[0]=255;
        new_bit[1]=0;
        int lowlay=0;
        if(((unsigned short*)new_bit)[0]==255){  //data layout test
            lowlay=1;
        }
        memset(new_bit,255,bit_size);
        if(bw==8||lowlay==0){
            for(int i=0;i<src_bit_size;i++) new_bit[i]=bccoo->bit[i];
        }
        else if(bw==16){
            for(int i=0;i<src_bit_size/2;i++){
                new_bit[i*2+1]=bccoo->bit[i*2];
                new_bit[i*2]=bccoo->bit[i*2+1];
            }
        }
        else if(bw==32){
            for(int i=0;i<src_bit_size/4;i++){
                new_bit[i*4+3]=bccoo->bit[i*4];
                new_bit[i*4+2]=bccoo->bit[i*4+1];
                new_bit[i*4+1]=bccoo->bit[i*4+2];
                new_bit[i*4]=bccoo->bit[i*4+3];
            }
        }
        create(clCxt,&(clbccoo->bit),bit_size);
        upload(clCxt,(void *)new_bit,clbccoo->bit,bit_size);
        free(new_bit);

        dataType *inter = new dataType[inter_size/sizeof(dataType)];
        for(int i=0;i<inter_size/sizeof(dataType);i++) inter[i]=CL_FLT_MAX;
        create(clCxt,&(clbccoo->inter),inter_size); 
        upload(clCxt,(void *)inter,clbccoo->inter,inter_size);

        create(clCxt,&(clbccoo->col),col_size); 
        transpose(clCxt,clbccoo->col,src_col,src_col_size,sizeof(dimType)*8,plan->cta,plan->workgroup,lt,tr);

        create(clCxt,&(clbccoo->res_entry),res_entry_size); 
        create(clCxt,&(clbccoo->para_scan),para_scan_size); 
        if(plan->col_delta==1)
            create(clCxt,&(clbccoo->col_delta),col_delta_size);
        getResEntryGpu(clCxt,clbccoo->bit,clbccoo->res_entry,clbccoo->col,
                          clbccoo->col_delta,clbccoo->para_scan,plan->workgroup,
                          plan->localthread,plan->cta,plan->bitwidth,plan->col_delta);
        create(clCxt,&(clbccoo->data),data_size); 
        transpose(clCxt,clbccoo->data,src_data,src_data_size,plan->block_width,plan->cta,plan->workgroup,lt,tr);
        if(plan->block_height>1){
            create(clCxt,&(clbccoo->data1),data_size); 
            transpose(clCxt,clbccoo->data1,src_data1,src_data_size,plan->block_width,plan->cta,plan->workgroup,lt,tr);
        }
        if(plan->block_height>2){
            create(clCxt,&(clbccoo->data2),data_size); 
            transpose(clCxt,clbccoo->data2,src_data2,src_data_size,plan->block_width,plan->cta,plan->workgroup,lt,tr);
        }
        if(plan->block_height>3){
            create(clCxt,&(clbccoo->data3),data_size); 
            transpose(clCxt,clbccoo->data3,src_data3,src_data_size,plan->block_width,plan->cta,plan->workgroup,lt,tr);
        }

#if defined ESTIMATE
        for(int tx=0;tx<=0;tx++){
        for(int co=0;co<=2;co++){
        for(int lg=0;lg<=0;lg++){
#else
        for(int tx=0;tx<=0;tx++){
        for(int co=0;co<=4;co++){
        for(int lg=0;lg<=1;lg++){
#endif
            int rg= gp - lg;
            if(tr==0&&co!=0) continue;
            if(co==0&&rg>1) continue;
            if(lg==0&&tr==0) continue;
            if(co==0&&(lt*(plan->block_height==3?4:plan->block_height)*plan->bitwidth>SH_MEM_SIZE/4))  continue;
            if(tr==1&&co!=0&&(rg==0||lg!=0)) continue;
            if((rg==0&&lg==0)||rg>4) continue;
#if defined PERF
            cout<<"tx:"<<tx<<" co:"<<co<<" rg:"<<rg<<" lg:"<<lg<<" wg:"<<plan->workgroup<<" >>>>> ";
#endif
            plan->registergroup = rg;
            plan->localmemgroup = lg;
            plan->coalesced = co;
            //plan->tx = tx;
            plan->tx = 0;
            cl_mem res_dev;
            create(clCxt,&res_dev,res_size);
            if(plan->tx==0)
                yaSpMVRun<dataType>(clCxt,clbccoo,vec_dev,res_dev,plan,TIMES,1);
            else 
                yaSpMVRun<dataType>(clCxt,clbccoo,vec_dev_tx,res_dev,plan,TIMES,1);
#if defined DEBUG
            dataType *res = new dataType[clbccoo->rows];
            download(clCxt,res_dev,(void *)res,clbccoo->rows*sizeof(dataType));
            check(cres,res,clbccoo->rows);
            delete[] res; 
#endif
            clReleaseMemObject(res_dev);
            if(timeRcd.totaltime < timeRcd.min_totaltime && timeRcd.totaltime > 0){
                timeRcd.min_totaltime = timeRcd.totaltime;
                best->localthread = plan->localthread; 
                best->registergroup = plan->registergroup;
                best->localmemgroup = plan->localmemgroup;
                best->cta = plan->cta;
                best->block_width = plan->block_width;
                best->block_height = plan->block_height;
                best->trans = plan->trans;
                best->tx = plan->tx;
                best->coalesced = plan->coalesced;
                best->slices = plan->slices;
                best->workgroup = plan->workgroup;
                best->col_delta = plan->col_delta;
                best->dimwidth = plan->dimwidth;
                best->bitwidth = plan->bitwidth;
            }
        }}}
        clReleaseMemObject(clbccoo->data);
        if(plan->block_height>1)
            clReleaseMemObject(clbccoo->data1);
        if(plan->block_height>2)
            clReleaseMemObject(clbccoo->data2);
        if(plan->block_height>3)
            clReleaseMemObject(clbccoo->data3);
        delete[] inter; 
        clReleaseMemObject(clbccoo->inter);
        clReleaseMemObject(clbccoo->para_scan);
        if(plan->col_delta == 1)
            clReleaseMemObject(clbccoo->col_delta);
        clReleaseMemObject(clbccoo->bit);
        clReleaseMemObject(clbccoo->res_entry);
        clReleaseMemObject(clbccoo->col);
    }}}}}
    clReleaseMemObject(vec_dev_tx);
    clReleaseMemObject(vec_dev);
    clReleaseMemObject(src_col);
    clReleaseMemObject(src_data);
    if(plan->block_height>1)
        clReleaseMemObject(src_data1);
    if(plan->block_height>2)
        clReleaseMemObject(src_data2);
    if(plan->block_height>3)
        clReleaseMemObject(src_data3);
    free(plan);
    return;
}

template<class dataType,class dimType,class bitType>
void yaSpMVbccoo2clbccoo(clContext *clCxt,BCCOO<dataType,dimType,bitType> *bccoo,CLBCCOO *clbccoo,Plan *plan)
{
    clbccoo->rows = bccoo->rows;
    clbccoo->cols = bccoo->cols;
    clbccoo->nnz = bccoo->nnz;
    clbccoo->block_number = bccoo->block_number;
    plan->workgroup = clbccoo->block_number/(plan->localthread*plan->cta);
    clbccoo->block_width = bccoo->block_width;
    clbccoo->block_height = bccoo->block_height;
    clbccoo->max_block_per_row = bccoo->max_block_per_row;
    clbccoo->slice_rows = bccoo->slice_rows;
    clbccoo->slices = bccoo->slices;
    int para_scan_size = (plan->workgroup)*sizeof(int);
    int inter_size = plan->workgroup*sizeof(dataType)*(bccoo->block_height>2?4:bccoo->block_height);
    int data_size = bccoo->block_width * bccoo->block_number*sizeof(dataType);
    int col_size = bccoo->block_number*sizeof(dimType);
    int bit_size = bccoo->block_number/8;
    int interres_size = bccoo->slice_rows*sizeof(dataType);
    int res_entry_size = (plan->workgroup * plan->localthread+1) * sizeof(int);
    int vec_size = bccoo->cols * sizeof(dataType);
    int res_size = bccoo->rows * sizeof(dataType);
    dataType *inter = new dataType[inter_size/sizeof(dataType)];
    int col_delta_size = bccoo->block_number * sizeof(short);
    
    create(clCxt,&(clbccoo->bit),bit_size);
    upload(clCxt,(void *)bccoo->bit,clbccoo->bit,bit_size);
    for(int i=0;i<inter_size/sizeof(dataType);i++) inter[i]=CL_FLT_MAX;
    create(clCxt,&(clbccoo->inter),inter_size);
    upload(clCxt,(void *)inter,clbccoo->inter,inter_size);

    create(clCxt,&(clbccoo->col),col_size);
    upload(clCxt,(void *)bccoo->col,clbccoo->col,col_size);
    create(clCxt,&(clbccoo->res_entry),res_entry_size); 
    create(clCxt,&(clbccoo->para_scan),para_scan_size); 
    if(plan->col_delta==1)
        create(clCxt,&(clbccoo->col_delta),col_delta_size);
    getResEntryGpu(clCxt,clbccoo->bit,clbccoo->res_entry,clbccoo->col,
              clbccoo->col_delta,clbccoo->para_scan,plan->workgroup,
              plan->localthread,plan->cta,plan->bitwidth,plan->col_delta);
    create(clCxt,&(clbccoo->data),data_size); 
    if(bccoo->block_height>1)  create(clCxt,&(clbccoo->data1),data_size); 
    if(bccoo->block_height>2)  create(clCxt,&(clbccoo->data2),data_size); 
    if(bccoo->block_height>3)  create(clCxt,&(clbccoo->data3),data_size); 
    upload(clCxt,(void *)bccoo->data,clbccoo->data,data_size);
    if(bccoo->block_height>1)
        upload(clCxt,(void *)(&(bccoo->data[data_size/sizeof(dataType)])),clbccoo->data1,data_size);
    if(bccoo->block_height>2)
        upload(clCxt,(void *)(&(bccoo->data[2*data_size/sizeof(dataType)])),clbccoo->data2,data_size);
    if(bccoo->block_height>3)
        upload(clCxt,(void *)(&(bccoo->data[3*data_size/sizeof(dataType)])),clbccoo->data3,data_size);
}

template<class dataType>
void yaSpMVmtx2clbccoo(clContext *clCxt,MTX<dataType> *mtx,CLBCCOO *clbccoo,Plan *best,int tune=1){
    int block_size[12][3],k=0;
#if defined DEBUG
    dataType *cres = new dataType[mtx->rows];
    memset(cres, 0, mtx->rows*sizeof(dataType));
    dataType *vec = new dataType[mtx->cols];
    for(int i=0;i<mtx->cols;i++) vec[i]=i;
    cpu_spmv<dataType>(mtx,vec,cres);
#endif
#if defined PERF
    cl_ulong cstart,cend;
    struct timeval vstart,vend;
    gettimeofday(&vstart,NULL);
#endif
    for(int bs=0;bs<12;bs++){
        block_size[bs][2]=CL_INT_MAX;
        block_size[bs][1]=0;
        block_size[bs][0]=0;
    }

    if(tune==1)
        footPrintSort(block_size,mtx);

    for(int bs=0;bs<4&&tune==1;bs++){
        int localthread = 64, cta = 8;
        if(block_size[bs][0]==0||block_size[bs][1]==0) continue;
        for(int slices = 1; slices <= 32; slices*=2){
            if(mtx->cols <= mtx->rows && slices != 1) continue;
            if((mtx->cols+block_size[bs][1]-1)/block_size[bs][1]>65535){
                BCCOO<dataType,unsigned int, unsigned char> bccoo;
                yaSpMVmtx2bccoo(mtx,block_size[bs][1],block_size[bs][0],
                           localthread,cta,0,slices,&bccoo);
#if defined DEBUG
                getPlan(clCxt,&bccoo,best,cres);
#else
                getPlan(clCxt,&bccoo,best);
#endif
                free(bccoo.bit);
                free(bccoo.col);
                free(bccoo.data);
            }
            else{
                BCCOO<dataType,unsigned short,unsigned char> bccoo;
                yaSpMVmtx2bccoo(mtx,block_size[bs][1],block_size[bs][0],
                            localthread,cta,0,slices,&bccoo);
#if defined DEBUG
                getPlan(clCxt,&bccoo,best,cres);
#else
                getPlan(clCxt,&bccoo,best);
#endif
                free(bccoo.bit);
                free(bccoo.col);
                free(bccoo.data);
            }
        }
    }

    if(best->dimwidth==sizeof(short)*8){
        if(best->bitwidth==sizeof(char)*8){
            BCCOO<dataType,unsigned short,unsigned char> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(short)*8){
            BCCOO<dataType,unsigned short,unsigned short> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(int)*8){
            BCCOO<dataType,unsigned short,unsigned int> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
    }
    if(best->dimwidth==sizeof(int)*8){
        if(best->bitwidth==sizeof(char)*8){
            BCCOO<dataType,unsigned int,unsigned char> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(short)*8){
            BCCOO<dataType,unsigned int,unsigned short> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(int)*8){
            BCCOO<dataType,unsigned int,unsigned int> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
    }
#if defined PERF
    gettimeofday(&vend,NULL);
    cstart=(cl_ulong)vstart.tv_sec*1000000 + (cl_ulong)vstart.tv_usec;
    cend=(cl_ulong)vend.tv_sec*1000000 + (cl_ulong)vend.tv_usec;
    cout<<"auto-tuning time:"<<cend - cstart<<endl;
#endif
}
