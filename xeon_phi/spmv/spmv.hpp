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
//    if(plan->col_delta==1)
//        args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->col_delta) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->bit) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->res_entry) ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->inter) ));
//    args.push_back( make_pair( sizeof(cl_mem) , (void *)&(clbccoo->para_scan) ));
    args.push_back( make_pair( sizeof(cl_int) , (void *)&plan->workgroup ));
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&vec_dev ));
//    cl_mem inter_res_dev;
//    int inter_res_size = clbccoo->slice_rows*sizeof(dataType);
//    if(clbccoo->slices!=1) {
//        create(clCxt,&inter_res_dev,inter_res_size); 
//        args.push_back( make_pair( sizeof(cl_mem) , (void *)&inter_res_dev ));
//    }
//    else
    args.push_back( make_pair( sizeof(cl_mem) , (void *)&res_dev ));
//    int registersize = plan->bitwidth*plan->registergroup;
//    int localmemsize = plan->bitwidth*plan->localmemgroup;
    char build_options[200];
//    if(plan->coalesced==0){
//        sprintf(build_options ,
//            "-D DIM_%d -D TRANS_%d -D TX_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_SIZE=%d -D NB_LOCAL_SIZE=%d -D NB_CTA_%d -D BIT_%d -D SEGSCAN_1 -D COL_COM_%d",
//            plan->dimwidth/8,plan->trans,plan->tx,plan->block_width,plan->block_height,
//            TEXTURE_WIDTH,plan->localthread,registersize,localmemsize,localmemsize,plan->bitwidth,
//            plan->col_delta);
//    }
//    else if(plan->trans==1){
//        sprintf(build_options ,
//            "-D DIM_%d -D TX_%d -D TRANS_%d -D NB_VEC_%d  -D BLOCK_HEIGHT_%d -D TX_WIDTH=%d -D NB_L%d -D NB_REG_GRP=%d -D BIT_%d -D SEGSCAN_1 -D CACHE_LEN=%d -D COL_COM_%d",
//            plan->dimwidth/8,plan->tx,plan->trans,plan->block_width,plan->block_height,
//            TEXTURE_WIDTH,plan->localthread,plan->registergroup,plan->bitwidth,
//            plan->coalesced*plan->localthread,plan->col_delta);
//    }
//    else
//        cout<<"The parameter of plan is unreasonable!"<<endl;
#if defined DEBUG
    //cout<<build_options<<endl;
#endif
    size_t globalthreads[3] = {plan->workgroup*1,1,1};
    //size_t globalthreads[3] = {240*5,1,1};
    size_t localthreads[3] = {1,1,1};
    cl_kernel kernel[1];
    vector<pair<size_t * ,size_t *> > threads;
    threads.push_back( make_pair( globalthreads, localthreads) );
//#if 1
    //kernel[0] = getKernel("spmv_strategy_a.cl","spmv_bccoo",args,build_options,clCxt);
//#else
    //kernel[0] = getKernel("spmv_strategy_b.cl","spmv_bccoo",args,build_options,clCxt);
//#endif
    kernel[0] = getKernel("spmv_strategy_a.cl","spmv_bccoo",args,build_options,clCxt);
    timeRcd.totaltime = 0;
    executeKernel(threads,kernel,clCxt,times,record);
    for(int i=0;i<threads.size();i++){
        cl_int ret = clReleaseKernel(kernel[0]);
//        if(ret != CL_SUCCESS) cout << "Failed to release kernel. error code:"<< ret  << endl;
    }
}

template<class dataType,class dimType,class bitType>
void yaSpMVbccoo2clbccoo(clContext *clCxt,BCCOO<dataType,dimType,bitType> *bccoo,CLBCCOO *clbccoo,Plan *plan)
{
    clbccoo->rows = bccoo->rows;
    clbccoo->cols = bccoo->cols;
    clbccoo->nnz = bccoo->nnz;
    clbccoo->block_number = bccoo->block_number;
    //plan->workgroup = clbccoo->block_number/(plan->localthread*plan->cta);
    plan->workgroup = clbccoo->block_number/(plan->cta);
    clbccoo->block_width = bccoo->block_width;
    clbccoo->block_height = bccoo->block_height;
    //clbccoo->max_block_per_row = bccoo->max_block_per_row;
    //clbccoo->slice_rows = bccoo->slice_rows;
    //clbccoo->slices = bccoo->slices;
    //int para_scan_size = (plan->workgroup)*sizeof(int);
    int inter_size = plan->workgroup*sizeof(dataType)*(bccoo->block_height>2?4:bccoo->block_height);
    int data_size = bccoo->block_width * bccoo->block_number*sizeof(dataType);
    int col_size = bccoo->block_number*sizeof(dimType);
    int bit_size = bccoo->block_number/8;
    int res_entry_size = (plan->workgroup+1) * sizeof(int);
    int vec_size = bccoo->cols * sizeof(dataType);
    int res_size = bccoo->rows * sizeof(dataType);
    
    dataType *inter = new dataType[inter_size/sizeof(dataType)];
    create(clCxt,&(clbccoo->bit),bit_size);
    upload(clCxt,(void *)bccoo->bit,clbccoo->bit,bit_size);
    for(int i=0;i<inter_size/sizeof(dataType);i++) inter[i]=CL_FLT_MAX;
    create(clCxt,&(clbccoo->inter),inter_size);
    upload(clCxt,(void *)inter,clbccoo->inter,inter_size);

    create(clCxt,&(clbccoo->col),col_size);
    upload(clCxt,(void *)bccoo->col,clbccoo->col,col_size);

    create(clCxt,&(clbccoo->res_entry),res_entry_size); 
    int * res_entry = (int *)malloc(res_entry_size); 
    getResEntryCpu(bccoo,res_entry,NULL,NULL, 1,plan->cta);
    upload(clCxt,(void *)res_entry,clbccoo->res_entry,res_entry_size);
/*
    getResEntryGpu(clCxt,clbccoo->bit,clbccoo->res_entry,clbccoo->col,
              clbccoo->col_delta,clbccoo->para_scan,plan->workgroup,
              plan->localthread,plan->cta,plan->bitwidth,plan->col_delta);
*/
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
void yaSpMVmtx2clbccoo(clContext *clCxt,MTX<dataType> *mtx,CLBCCOO *clbccoo,Plan *best,int tune=0){
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

//    if(tune==1)
//        footPrintSort(block_size,mtx);

    if(best->dimwidth==sizeof(short)*8){
        if(best->bitwidth==sizeof(char)*8){
            BCCOO<dataType,unsigned short,unsigned char> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    1,best->cta,0,1,&bccoo);
            //yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
            //        best->localthread,best->cta,best->trans,best->slices,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(short)*8){
            BCCOO<dataType,unsigned short,unsigned short> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    1,best->cta,0,1,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(int)*8){
            BCCOO<dataType,unsigned short,unsigned int> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    1,best->cta,0,1,&bccoo);
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
                    1,best->cta,0,1,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(short)*8){
            BCCOO<dataType,unsigned int,unsigned short> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    1,best->cta,0,1,&bccoo);
            yaSpMVbccoo2clbccoo(clCxt,&bccoo,clbccoo,best);
            free(bccoo.bit);
            free(bccoo.col);
            free(bccoo.data);
        }
        if(best->bitwidth==sizeof(int)*8){
            BCCOO<dataType,unsigned int,unsigned int> bccoo;
            yaSpMVmtx2bccoo(mtx,best->block_width,best->block_height,
                    1,best->cta,0,1,&bccoo);
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
