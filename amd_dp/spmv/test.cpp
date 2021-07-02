#include "spmv.hpp"

int main(int argc, char* argv[]){
    if (argc < 2)
    {
        cout<<"Usage: ./spmv sparse.mtx [optimal.cfg]"<<endl;
        return 0;
    }
    char* filename = argv[1];
    FILE* infile = fopen(filename, "r");
    clContext clCxt;
    getClContext(&clCxt);
    timeRcd.min_totaltime=10000000;
    Plan best;
    CLBCCOO clbccoo;
    MTX<double> mtx;
    fileToMtx<double>(filename,&mtx);
    cl_mem vec_dev,res_dev;
    double *vec = new double[mtx.cols];
    for(int i=0;i<mtx.cols;i++) vec[i]=i;
    int tune = 1;
    if(argc == 3){
        FILE* infile_1 = fopen(argv[2], "r");
        fscanf(infile_1, "%d", &best.dimwidth);
        fscanf(infile_1, "%d", &best.bitwidth);
        fscanf(infile_1, "%d", &best.block_width);
        fscanf(infile_1, "%d", &best.block_height);
        fscanf(infile_1, "%d", &best.localthread);
        fscanf(infile_1, "%d", &best.slices);
        fscanf(infile_1, "%d", &best.trans);
        fscanf(infile_1, "%d", &best.col_delta);
        fscanf(infile_1, "%d", &best.cta);
        fscanf(infile_1, "%d", &best.tx);
        fscanf(infile_1, "%d", &best.coalesced);
        fscanf(infile_1, "%d", &best.registergroup);
        fscanf(infile_1, "%d", &best.localmemgroup);
        fclose(infile_1);
        tune = 0;
    }
    else
        generateProgramCache<double>(&clCxt); //only support double
    
    yaSpMVmtx2clbccoo<double>(&clCxt,&mtx,&clbccoo,&best,tune);
    
    double *cres = new double[mtx.rows];
    cpu_spmv<double>(&mtx,vec,cres);
 
    create(&clCxt,&res_dev,clbccoo.rows*sizeof(double));
    if(best.tx == 1)
        getTexture(&clCxt, vec_dev, vec, clbccoo.cols*sizeof(double), best.block_width,TEXTURE_WIDTH);
    else{
        create(&clCxt,&vec_dev,clbccoo.cols*sizeof(double));
        upload(&clCxt,(void *)vec,vec_dev,clbccoo.cols*sizeof(double));
    }
    cl_ulong cstart,cend;
    struct timeval vstart,vend;
    gettimeofday(&vstart,NULL);
    int run_times=10000;
    yaSpMVRun<double>(&clCxt, &clbccoo,vec_dev,res_dev,&best,run_times);
    clFinish(clCxt.command_queue);
    gettimeofday(&vend,NULL);

    double *res = new double[clbccoo.rows];
    download(&clCxt,res_dev,(void *)res,clbccoo.rows*sizeof(double));
    check(cres,res,clbccoo.rows);
    delete[] res;

    cstart=(cl_ulong)vstart.tv_sec*1000000 + (cl_ulong)vstart.tv_usec;
    cend=(cl_ulong)vend.tv_sec*1000000 + (cl_ulong)vend.tv_usec;
    double t = (double)(cend -cstart)/1000/run_times;
    cout<<"Best Plan: dim:"<<best.dimwidth<<" bw:"<<best.bitwidth<<" wid:"<<best.block_width<<" hei:"<<best.block_height;
    cout<<" lt:"<<best.localthread<<" sli:"<<best.slices<<" tr:"<<best.trans <<" cd:"<<best.col_delta<<" cta:"<<best.cta;
    cout<<" tx:"<<best.tx<<" co:"<<best.coalesced<<" rg:"<<best.registergroup<<" lg:"<<best.localmemgroup<<endl;
    
    cout<<"Program Execute time:"<<t<<" milliseconds."<<endl;
    return 0;
}
