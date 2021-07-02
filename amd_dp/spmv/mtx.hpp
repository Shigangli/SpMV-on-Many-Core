template<class dataType>
bool if_sorted_coo(MTX<dataType>* mtx)
{
    int nnz = mtx->nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mtx->row[i] > mtx->row[i+1]) || (mtx->row[i] == mtx->row[i+1] && mtx->col[i] > mtx->col[i+1]))
            return false;
    }
    return true;
}

template<class dataType>
bool sort_coo(MTX<dataType>* mtx)
{

    int i = 0;
    int beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    int pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mtx->nnz;
    while (i>=0) 
    {
        L=beg[i];
        if (end[i] - 1 > end[i])
            R = end[i];
        else
            R = end[i] - 1;
        if (L<R) 
        {
            int middle = (L+R)/2;
            pivrow=mtx->row[middle]; 
            pivcol=mtx->col[middle];
            pivdata=mtx->data[middle];
            mtx->row[middle] = mtx->row[L];
            mtx->col[middle] = mtx->col[L];
            mtx->data[middle] = mtx->data[L];
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            if (i==MAX_LEVELS-1) 
                return false;
            while (L<R) 
            {
                while (((mtx->row[R] > pivrow) || 
                            (mtx->row[R] == pivrow && mtx->col[R] > pivcol)) 
                        && L<R) 
                    R--; 
                if (L<R) 
                {
                    mtx->row[L] = mtx->row[R];
                    mtx->col[L] = mtx->col[R];
                    mtx->data[L] = mtx->data[R];
                    L++;
                }
                while (((mtx->row[L] < pivrow) || 
                            (mtx->row[L] == pivrow && mtx->col[L] < pivcol)) 
                        && L<R) 
                    L++; 
                if (L<R) 
                {
                    mtx->row[R] = mtx->row[L];
                    mtx->col[R] = mtx->col[L];
                    mtx->data[R] = mtx->data[L];
                    R--;
                }
            }
            mtx->row[L] = pivrow;
            mtx->col[L] = pivcol;
            mtx->data[L] = pivdata;
            beg[i+1]=L+1; 
            end[i+1]=end[i]; 
            end[i++]=L; 
        }
        else 
        {
            i--; 
        }
    }
    return true;
}

template<class dataType>
void fileToMtx(char* filename,MTX<dataType> *mtx)
{
    FILE* infile = fopen(filename, "r");
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    bool ifreal = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;
    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
        fscanf(infile, "%s", tmpstr);
        if (tmpstr[0] != '%')
        {
            height = atoi(tmpstr);
            break;
        }
        fgets(tmpline, 1025, infile);
    }

    fscanf(infile, "%d %d", &width, &nnz);
    mtx->rows = height;
    mtx->cols = width;
    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    dataType* data = (double*)malloc(sizeof(double)*nnz);
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = i;
        if (ifreal)
        {
            double dbldata = 0.0f;
            fscanf(infile, "%lf", &dbldata);
            data[i] = (dataType)dbldata;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    
    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mtx->nnz = newnnz;
        mtx->row = (int*)malloc(sizeof(int)*newnnz);
        mtx->col = (int*)malloc(sizeof(int)*newnnz);
        mtx->data = (dataType*)malloc(sizeof(dataType)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mtx->row[matid] = rows[i];
            mtx->col[matid] = cols[i];
            mtx->data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mtx->row[matid] = cols[i];
                mtx->col[matid] = rows[i];
                mtx->data[matid] = data[i];
                matid++;
            }
        }
        if(matid != newnnz){
            cout<<"Error!"<<endl;
        }
        bool tmp = sort_coo<dataType>(mtx);
        if(tmp != true){
            cout<<"Error!"<<endl;
        }
    }
    else
    {
        mtx->nnz = nnz;
        mtx->row = (int*)malloc(sizeof(int)*nnz);
        mtx->col = (int*)malloc(sizeof(int)*nnz);
        mtx->data = (dataType*)malloc(sizeof(dataType)*nnz);
        memcpy(mtx->row, rows, sizeof(int)*nnz);
        memcpy(mtx->col, cols, sizeof(int)*nnz);
        memcpy(mtx->data, data, sizeof(dataType)*nnz);
        if (!if_sorted_coo(mtx))
            sort_coo<dataType>(mtx);
        bool tmp=if_sorted_coo<dataType>(mtx);
        if(tmp != true){
            cout<<"Error!"<<endl;
        }
    }
    fclose(infile);
    free(rows);
    free(cols);
    free(data);
    return;    
}

template<class dataType>
void printMtx(MTX<dataType> *mtx)
{
    cout<<"rows:"<<mtx->rows<<"  cols:"<<mtx->cols<<"  non zeros:"<<mtx->nnz<<endl;
    for(int i=0;i<mtx->nnz;i++){
       cout<<mtx->col[i]<<"  "<<mtx->row[i]<<"  "<<mtx->data[i]<<endl;
    }
}

