template<class dataType>
int getBlock(MTX<dataType> *mtx,int l, int block_width, dataType *dst)
{
    int tail = mtx->col[l]%block_width,len=1;
    if(dst!=NULL)
        for(int i=0;i<block_width;i++) 
            dst[i]=0;
    if(dst!=NULL)
        dst[tail]=mtx->data[l];
    for(int i=1;i<block_width-tail;i++)
        if((mtx->col[l+i]<mtx->col[l]+block_width-tail)
            &&(mtx->row[l+i]==mtx->row[l])){
            if(dst!=NULL){
                dst[mtx->col[l+i]%block_width] = mtx->data[l+i];
            }
            len++;
        }  
    return len;
}

template<class dataType,class dimType,class bitType>
int getBlockRow(MTX<dataType> *mtx,int slice_rows,BCCOO<dataType,dimType,bitType> *bccoo,
                int row,int *hist, int*e_hist,int block_width,int block_height,int start,
                int aligned_width=0,int elem_per_thread=1)
{
    unsigned int min_col,dim_max;
    int m=0,count=0;
    min_col=CL_UINT_MAX;
    dim_max=CL_UINT_MAX;
    for(int i=0;i<block_height;i++){
        if(row+i<slice_rows&&hist[row+i]!=-1){
            if(mtx->col[hist[row+i]]<min_col){
                min_col = mtx->col[hist[row+i]];
            }
        }
    }
    if(min_col==dim_max){
        if(bccoo!=NULL){
            for(int i=0;i<bccoo->block_height;i++){
                for(int j=0;j<bccoo->block_width;j++){
                    int pos = start%elem_per_thread*aligned_width+start/elem_per_thread;
                    bccoo->data[bccoo->block_number*bccoo->block_width*i + pos*bccoo->block_width + j]=0;
                }
            }
            bccoo->col[start%elem_per_thread*aligned_width+start/elem_per_thread]=0;
        }
        return 1;
    }
    int *loc = new int[block_height];
    int *end = new int[block_height];
    for(int i=0;i<block_height;i++){
        loc[i]=-1;
        end[i]=-1;
        if(row+i<slice_rows){
            loc[i]=hist[row+i];
            end[i]=e_hist[row+i];
        }
    }
    int *blen = new int[block_height];
    while(true){
        min_col=dim_max;
        for(int i=0;i<block_height;i++){
            if(row+i<slice_rows&&loc[i]!=-1&&loc[i]<=end[i]){
                if(mtx->col[loc[i]]<min_col){
                    min_col = mtx->col[loc[i]];
                    m=i;
                }
            }
        }
        for(int i=0;i<block_height;i++){
            blen[i]=0;
        }
        if(bccoo!=NULL){
            int pos = (count+start)%elem_per_thread*aligned_width+(count+start)/elem_per_thread;
            blen[m] = getBlock<dataType>(mtx,loc[m],block_width,
                     &(bccoo->data[bccoo->block_number*bccoo->block_width*m + pos*bccoo->block_width]));
        }
        else{
            blen[m] = getBlock<dataType>(mtx,loc[m],block_width,NULL);
        }
        if(bccoo!=NULL){
            bccoo->col[(count+start)%elem_per_thread*aligned_width+(count+start)/elem_per_thread]=
                     (dimType)(mtx->col[loc[m]]/bccoo->block_width);
        }
        for(int i=0;i<block_height;i++){
            if(i!=m){
                if(loc[i]!=-1&&(mtx->col[loc[i]]<(((mtx->col[loc[m]]+block_width)/block_width)*block_width))
                    &&(mtx->row[loc[i]]==mtx->row[loc[m]]+i-m)){
                    if(bccoo!=NULL){
                        int pos = (count+start)%elem_per_thread*aligned_width+(count+start)/elem_per_thread;
                        blen[i] = getBlock<dataType>(mtx,loc[i],block_width,
                              &(bccoo->data[bccoo->block_number*bccoo->block_width*i + pos*bccoo->block_width]));
                    }
                    else{
                        blen[i] = getBlock<dataType>(mtx,loc[i],block_width,NULL);
                    }
                }
                loc[i]+=blen[i];
            }
        }
        loc[m]+=blen[m];
        count++;
        int con=0;
        for(int i=0;i<block_height;i++){
            if(loc[i]!=-1&&loc[i]<=end[i]){
                con=1;
            }
        }
        if(con==0)
            break;
    }
    delete[] loc;
    delete[] blen;
    delete[] end;
    return count;
}

int getBlockNumber(int *row,int *col,int cols,int nnz,int block_width,int block_height)
{
    unsigned int * hash=new unsigned int[cols/block_width+1];
    memset(hash,255,(cols+block_width-1)/block_width*4);
    int count = 0;
    for(int i=0;i<nnz;i++){
        if(hash[col[i]/block_width]!=row[i]/block_height) count++;
        hash[col[i]/block_width]=row[i]/block_height;
    }
    delete[] hash;
    return count;
}

template<class dataType,class dimType,class bitType>
int yaSpMVmtx2bccoo(MTX<dataType> *mtx,int block_width,int block_height,int workgroup_size,
     int elem_per_thread,int trans,int slices, BCCOO<dataType,dimType,bitType> *bccoo)
{
    int bitwidth=sizeof(bitType)*8,bitwidth_1=bitwidth-1;
    int slice_rows=(mtx->rows+block_height-1)/block_height*block_height*slices;
    int slice_width = mtx->cols / (block_width*slices) * block_width;
    int *hist = new int[slice_rows];
    int *e_hist = new int[slice_rows];
    for(int i=0;i< slice_rows;i++){
        hist[i]=-1;
        e_hist[i]=-1;
    }
    int row=mtx->row[0],row2=mtx->row[0];
    int ii=0,s_row=-1;
    for(int j=0;j<slices;j++){
        int s_col = slice_width*j,e_col=(j==slices-1)?mtx->cols:slice_width*(j+1);
        for(int i=0;i < mtx->nnz;i++){
            if(mtx->col[i]>=s_col&&mtx->col[i]<e_col){
                row2 = mtx->row[i] + slice_rows/slices*j;
                if(row2!=row){
                    hist[row2]=i;
                    if(s_row!=-1) 
                        e_hist[row]=ii;
                }
                ii=i;
                row=row2;
                if(s_row==-1) s_row=row;
            }
        }
    }
    hist[s_row]=0;
    e_hist[row]=ii;
    int count=0,max_block_per_row=0,block_per_row=0;
    for(int j=0;j<slices;j++){
        for(int i=0;i<slice_rows/slices;i=i+block_height){
            block_per_row=getBlockRow<dataType,dimType,bitType>(mtx,slice_rows/slices*(j+1),
                     NULL,i+slice_rows/slices*j,hist,e_hist,block_width,block_height,count);
            max_block_per_row = block_per_row > max_block_per_row ? block_per_row : max_block_per_row;
            count+=block_per_row;
        }
    }
    int block_number = count - count % 
                          (workgroup_size * elem_per_thread) + (workgroup_size * elem_per_thread);
    if(bccoo==NULL)
        return block_number;

    bccoo->rows = mtx->rows;
    bccoo->cols = mtx->cols;
    bccoo->nnz = mtx->nnz;
    bccoo->block_width = block_width;
    bccoo->block_height = block_height;
    bccoo->slice_rows=slice_rows;
    bccoo->slices=slices;
    bccoo->max_block_per_row = max_block_per_row;
    bccoo->block_number = block_number;
    bccoo->data = (dataType *)malloc(sizeof(dataType)*bccoo->block_width*bccoo->block_height*bccoo->block_number);
    bccoo->col = (dimType *)malloc(sizeof(dimType)*bccoo->block_number);
    bccoo->bit = (bitType *)malloc(bccoo->block_number/8);
    memset(bccoo->data,0,sizeof(dataType)*bccoo->block_width*bccoo->block_height*bccoo->block_number);
    memset(bccoo->col,0,sizeof(dimType)*bccoo->block_number);
    memset(bccoo->bit,255,bccoo->block_number/8);
    count=0;
    int aligned_width=bccoo->block_number/elem_per_thread;
    for(int j=0;j<bccoo->slices;j++){
        for(int i=0;i<bccoo->slice_rows/bccoo->slices;i=i+block_height){
            if(trans==1)
                count+=getBlockRow<dataType,dimType,bitType>(mtx,bccoo->slice_rows/bccoo->slices*(j+1),
                            bccoo,i+bccoo->slice_rows/bccoo->slices*j,hist,e_hist,
                            block_width,block_height,count,aligned_width,elem_per_thread);
            else
                count+=getBlockRow<dataType,dimType,bitType>(mtx,bccoo->slice_rows/bccoo->slices*(j+1),
                            bccoo,i+bccoo->slice_rows/bccoo->slices*j,hist,e_hist,block_width,block_height,count);
            bitType bit=0;
            bit = 1<<(bitwidth_1 - (count-1)%bitwidth);
            bccoo->bit[(count-1)/bitwidth]=
            bccoo->bit[(count-1)/bitwidth]^bit;
        }
    }
    delete[] hist;
    delete[] e_hist;
    return block_number;
}

template<class dataType,class dimType,class bitType>
void printBCCOO(BCCOO<dataType,dimType,bitType> *bccoo)
{
    int bitwidth=sizeof(bitType)*8,bitwidth_1=bitwidth-1;
    cout<<"rows:"<<bccoo->rows<<"  cols:"<<bccoo->cols<<" nnz:"<<bccoo->nnz<<"  block number:"<<
       bccoo->block_number<< " data length:"<<bccoo->block_width*bccoo->block_number<<endl;
    cout<<"BCCOO-bit:"<<endl;
    for(int i=0;i<bccoo->block_number;){
        cout<<"line:"<<i/32<<"  ";
        for(int j=0;j<32&&i<bccoo->block_number;j++,i++){
           bitType bit = bccoo->bit[i/(bitwidth)],mask = 1<<(bitwidth_1-i%(bitwidth));
           bit = bit & mask;
           cout<<" "<<(bit>0);
        }
        cout<<endl;
    }
    cout<<"BCCOO-col:"<<endl;
    for(int i=0;i<bccoo->block_number;){
        for(int j=0;j<32&&i<bccoo->block_number;j++,i++)
           cout<<" "<<bccoo->col[i];
        cout<<endl;
    }
    cout<<"BCCOO-data:"<<endl;
    for(int i=0;i<bccoo->block_number*bccoo->block_height;){
        for(int j=0;j<8&&i<bccoo->block_number*bccoo->block_height;j++,i++)
           for(int k=0;k<bccoo->block_width;k++)
               cout<<" "<<bccoo->data[i*bccoo->block_width + k];
        cout<<endl;
    }
}

template<class dataType,class dimType,class bitType>
void getResEntryCpu(BCCOO<dataType,dimType,bitType> *bccoo,int *res_entry,
                       short *col_delta,int *para_scan,int workgroup_size,int cta_size){
    int bitwidth=sizeof(bitType)*8,bitwidth_1=bitwidth-1,i_max=0;
    int seg=0;
    res_entry[0] = 0;
    if(bitwidth==8) i_max=255;
    if(bitwidth==16) i_max=65535;
    if(bitwidth==32) i_max=CL_UINT_MAX;
    for(int i=1;i<bccoo->block_number/(cta_size);i++)
    {
        int sum=0;
        for(int j=0;j<cta_size/bitwidth;j++){
            bitType mask = 1<<bitwidth_1;
            bitType bit = bccoo->bit[(i-1)*(cta_size/bitwidth)+j];
            for(int k=0;k<bitwidth;k++,mask>>=1){
                sum += (bit&mask)>0?0:1;
            }
        }
        res_entry[i]=sum + res_entry[i-1];
    }
    seg=0;
    for(int i=0;i<bccoo->block_number/cta_size;i++)
    {
        for(int j=0;j<cta_size/bitwidth;j++)
            if(bccoo->bit[i*(cta_size/bitwidth)+j]==i_max) seg=1;
        if(i%workgroup_size==workgroup_size-1){
            para_scan[i/workgroup_size]=seg;
            seg=0;
        }
        if(col_delta!=NULL){
            col_delta[i] = -1;
            for(int j=1;j<cta_size;j++){
                int diff = bccoo->col[i + j*(bccoo->block_number/cta_size)] - 
                           bccoo->col[i + (j-1) * (bccoo->block_number/cta_size)];
                if( diff > 32767 || diff < -32767 ){
                    col_delta[i + j*(bccoo->block_number/cta_size)] = -1;
                }
                else{
                    col_delta[i + j*(bccoo->block_number/cta_size)] = (short)diff;
                }
            }
        }
    }
    res_entry[bccoo->block_number/cta_size]= res_entry[bccoo->block_number/cta_size-1];
}
