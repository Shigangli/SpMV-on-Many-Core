template<class dataType>
void cpu_spmv(MTX<dataType> *mtx,dataType *vec,dataType *res)
{
    int row=mtx->row[0],row2;
    dataType sum=0;
    for(int i=0;i<mtx->nnz;i++){
        row2=mtx->row[i];
        if(row2!=row){
           res[row] = sum;
           sum=0;
        }
        row=row2;
        sum += mtx->data[i]*vec[mtx->col[i]];
    }
    res[row2]=sum;
}

template<class dataType>
void check(dataType *cres,dataType *res,int len)
{
    int count=0;
    dataType max=0.0,ratio=0.0;
    for(int i=0;i<len;i++){
        dataType minus = cres[i]-res[i];
        if(minus<0) minus=0-minus;
        dataType v=cres[i]>0?cres[i]:0-cres[i];
        if(minus>v/10000){
            count++;
            if(count<10)
                cout<<"Error: cpu "<<cres[i]<<" gpu "<<res[i]<<" line "<<i<<endl;
            if(v==0) v++;
            ratio = ratio < minus/v ? minus/v:ratio;
            max = max < minus ? minus : max;
        }
    }
    //if(max!=0)
        cout<<"Max diff:"<<max<<"  ratio:"<<ratio<<"  count:"<<count<<endl;
}

