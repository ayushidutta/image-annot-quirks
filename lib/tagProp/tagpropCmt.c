/*
 * =============================================================
 * tagpropDistC.c
 *
 * Computes log-likelihood and gradient for sigmoidal distance based model
 *
 *
 * This is a MEX-file for MATLAB.
 * JJV, March 2009, INRIA
 * =============================================================
 */

#include "mex.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

#define DB  0
#define epsil (1e-3)
#define LogEps (1e-15)


int count_cpu() {
    cpu_set_t set;
    sched_getaffinity(0, sizeof(cpu_set_t), &set);
    int i, count=0;
    for(i=0;i<CPU_SETSIZE;i++)
        if(CPU_ISSET(i, &set)) count++;
    return count;
}


void SoftMax( double *P, int K){
    int k;
    double MaxEnergy=P[0], EnergySum=0;
    
    for (k=0 ; k<K ; k++)
        if (P[k] > MaxEnergy)
            MaxEnergy = P[k];
        
    for (k=0;k<K;k++){
        P[k]     = exp( P[k] - MaxEnergy );
        EnergySum += P[k];
    }
    
    for (k=0;k<K;k++)
        P[k]  /= EnergySum;
}

double Sigmoid( double X){
    double result;
    
    result = 1 / ( 1 + exp(-X) );
    return result;
}


typedef struct {
    pthread_mutex_t mutex;
    int id;
    int Ncpu;
    int I;
    int K;
    int D;
    int W;
    int *Offsets;
    double *LC;
    double *AN;
    double *AW;
    double *NN;
    double *ND;
    double *A;
    double *B;
    double *f;
    double *g;    
} context_t;

static void *threadFunc(void *threadarg) {
    context_t *context = threadarg;
    
    int D=context->D, K=context->K, I=context->I, W=context->W;
    int *Offsets=context->Offsets;
    double *LC=context->LC, *AN=context->AN, *AW=context->AW, *NN=context->NN, *ND=context->ND, *A=context->A, *B=context->B;    
    double f=0, *g;    
    int id, i, Ncpu, KD=K*D;
    double  *Pr, *LLH, *Ed, *X;    
    char          *model, *mode, *weights;
    int           DoSigmoid = (context->A!=NULL);
    int           DoPredict = (context->g==NULL);
    int           RankBased = (context->ND==NULL);
    double  PriorSum =0, MaxEnergy = 0;
    
    pthread_mutex_lock( &context->mutex );
    id = context->id++;
    pthread_mutex_unlock( &context->mutex );
    Ncpu = context->Ncpu;
    
    model   = DoSigmoid ? "sigmoid": "linear";
    mode    = DoPredict ? "predict": "f + gr";
    weights = RankBased ? "Rank": "Distance";
    
    if (DB) printf("--> Thread %d/%d:  %s, %s, %s,  I=%d, W=%d, K=%d, D=%d\n", id, Ncpu, weights, model, mode, I, W, K, D);
    
    g       = malloc( D   * sizeof(double) );        
    Pr      = malloc( K   * sizeof(double) );
    LLH     = malloc( W   * sizeof(double) );
    Ed      = malloc( D   * sizeof(double) );
    X       = malloc( W   * sizeof(double) );
    
    memset(g, 0, D  *sizeof(double));    
    
    if (RankBased){/* pre-compute prior weight for neighbors */
        int k;
        for (k=0;k<K;k++)
            Pr[k] = -LC[k];        
        SoftMax(Pr,K);        
        memcpy(Ed, Pr, D*sizeof(double) );
    }
    
    for (i=id;i<I;i+=Ncpu) {
        double WeightSum = 0;
        double EdAccu, SumLLH=0;
        int    d, k, j, w, NNj, count, next_w;
        const int iW=i*W;
        
        if (DB>1) printf("--> Thread %d, image %d ", id, i);
    
        if (!RankBased){ /* compute prior weight for neighbors based on distances */
            memset(Pr, 0, K  *sizeof(double));
            memset(Ed, 0, D  *sizeof(double));
            
            j = i*KD; /* compute neighbor weights */
            for (k=0;k<K;k++)            
                for (d=0;d<D;d++)
                    Pr[k] -= LC[d] * ND[j++];            
            SoftMax(Pr,K);
            
            j = i*KD; /* compute expected dist */
            for (k=0;k<K;k++)           
                for (d=0;d<D;d++)
                    Ed[d] += Pr[k]*ND[j++];            
        }
                
        memset(X ,  0, W  *sizeof(double));
        j =i*K;
        for (k=0;k<K;k++){/* compute linear predictions for all words*/
            int    Nik = NN[j++]; /* index of neighbor k */
            int    OSk = Nik==0 ? 0 : Offsets[Nik-1];
                        
            for( count=OSk; count<Offsets[Nik]; count++ )                               
                X[(int) AN[count]] += Pr[k];            
        }   
        
        memset(LLH, 0, W  *sizeof(double));
        for (w=0;w<W;w++){ /* make offset from zero and one, useless in sigmoid model */
            X[w]   = epsil/2 + (1-epsil)*X[w];
            LLH[w] = X[w];
            if (DoSigmoid)
                LLH[w]  = Sigmoid( X[w]*A[w] + B[w] );                            
        }
        
        if (DoPredict){ /* store probability for tag presences */
            pthread_mutex_lock( &context->mutex );
            for (w=0;w<W;w++)
                context->f[iW+w] = LLH[w];
            pthread_mutex_unlock( &context->mutex );}
        
        if (!DoPredict){/*  compute function and gradient */
            j=iW;
            EdAccu = 0;
            count = i==0 ? 0 : Offsets[i-1];
            next_w=count<Offsets[i] ? AN[count] : -1;
            for (w=0;w<W;w++){ /* compute likelihood of annotation */
                double weight = AW[j++];
                double y2=-1; /* tag absence / presence */
                
                if (next_w==w){
                    y2 = 1;
                    count++;
                    next_w=count<Offsets[i] ? AN[count] : -1;}
                else
                    LLH[w] = 1-LLH[w];    /* likelihood of absence needed */
                
                f   -= weight * log(LLH[w]+LogEps);  /* increment log-likelihood */
                
                if (DoSigmoid){
                    LLH[w]  = weight * A[w] * (1-LLH[w]) * y2; /* weight needed for gradient */
                    EdAccu += X[w]*LLH[w];}
                else{
                    LLH[w]  = weight / LLH[w];                                       
                    EdAccu += weight;}
                SumLLH += LLH[w];
            }                    
            
            NNj =i*K;
            j = i*KD;
            for (k=0;k<K;k++){/* compute remaining gradient term*/
                int    Nik  = NN[NNj++];
                int    w1   = Nik==0 ? 0 : Offsets[Nik-1];
                int    w2   = Offsets[Nik];
                double accu = (DoSigmoid) ? 0 : SumLLH;
                                
                if (DoSigmoid)
                    for(w=w1;w<w2;w++)
                        accu += LLH[(int) AN[w]];
                else{
                    int v;
                    int v1 = i==0 ? 0 : Offsets[i-1];                                        
                    int v2 = Offsets[i];
                    
                    for(w=w1;w<w2;w++)                        
                        accu -= LLH[ (int) AN[w] ];
                
                    for(v=v1;v<v2;v++)
                        accu -= LLH[ (int) AN[v] ];
                
                    for(v=v1;v<v2;v++)                                                          
                        for(w=w1;w<w2;w++)
                            if ( AN[v] == AN[w] )
                                accu += 2*LLH[ (int) AN[v] ];                    
                }
                
                accu  = (epsil/2) * SumLLH + (1-epsil) * accu;                
                accu *= Pr[k];
                
                if (RankBased)
                    g[k] += accu;
                else
                    for (d=0;d<D;d++)
                        g[d] +=  accu * ND[j++];
            }            
            
            for (d=0;d<D;d++) 
                g[d] -= EdAccu * Ed[d];
        }
    }
    if (DB) printf("--> Thread %d finished processing \n", id);
    
    if (!DoPredict){ /* increment log-lik and gradient with results from this thread */
        pthread_mutex_lock( &context->mutex );
        context->f[0]+=f;
        for (i=0;i<D;i++)
            context->g[i]+=g[i];
        pthread_mutex_unlock( &context->mutex );
    }
                
    free(g      );
    free(Pr     );
    free(LLH    );
    free(Ed     );
    free(X      );
    
    return NULL;
}


void TagPropDistSigmoid( double *LC, double *AN, int *Offsets, double *AW, double *NN, double *ND, double *A, double *B, double *f, double *g, int D, int I, int W, int K) {
    int           i, Ncpu=count_cpu();
    pthread_t     pth[Ncpu]; /* this is our thread identifier*/
    context_t     context;        
    int           DoPredict = (g==NULL);
        
    if (DB) printf("--> TagProp, I=%d, W=%d, K=%d, D=%d, Ncpu=%d\n", I, W, K, D, Ncpu);
    
    if (DoPredict)                           /* initialize outputs */
        memset(f,    0, I*W*sizeof(double)); /* tag predictions    */
    else{
        f[0] = 0;   /* log-likelihood and gradient*/
        memset(g,    0,   D*sizeof(double));   
    }
        
    pthread_mutex_init( &context.mutex, NULL);
    context.id      = 0;
    context.Ncpu    = Ncpu;
    context.I       = I;
    context.K       = K;
    context.D       = D;
    context.W       = W;
    context.Offsets = Offsets;
    context.LC      = LC;
    context.AN      = AN;
    context.AW      = AW;
    context.NN      = NN;
    context.ND      = ND;
    context.A       = A;
    context.B       = B;
    context.f       = f;
    context.g       = g;                   
        
    for (i=0;i<Ncpu;i++) /* Create worker threads */
        pthread_create(&pth[i], NULL, &threadFunc,  &context);    
    for (i=0;i<Ncpu;i++) /* wait for threads to finish */
        pthread_join(pth[i], NULL );        
}


/* The gateway routine */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double  *LC, *AW, *ND, *f, *g=NULL, *AN, *NN, *A=NULL, *B=NULL, *NW;
    int      I,  K,  W, D, *Offsets, AnnIms, i;
    
    if ((nrhs != 8) && (nrhs !=6))    mexErrMsgTxt("Six or eight inputs required."); /* differentiate between linear and sigmoid models */
    if ((nlhs != 1) && (nlhs !=2))    mexErrMsgTxt("One or two outputs  required."); /* differentiate between function + grad evaluation, or predictions */
    
    /* Create pointers to potentials. */
    LC       = mxGetPr(prhs[0]); /* distance combination weights OR rank based weights */
    AN       = mxGetPr(prhs[1]); /* sparse annotations */
    NW       = mxGetPr(prhs[2]); /* number of words per image */
    AW       = mxGetPr(prhs[3]); /* annotation weights */
    NN       = mxGetPr(prhs[4]); /* nearest neighbor indices */
    ND       = mxGetPr(prhs[5]); /* nearest neighbor distances */
    
    if (mxGetNumberOfElements(prhs[5])==0)
        ND = NULL; /* for rank based model, the distance matrix is empty */        
    
    if (nrhs==8){
        A        = mxGetPr(prhs[6]); /* alpha parameters */
        B        = mxGetPr(prhs[7]);} /* beta  parameters */               
    
    D       = mxGetNumberOfElements(prhs[0]);  /* number of parameters */
    I       = mxGetN(prhs[4]);  /* number of images    */
    K       = mxGetM(prhs[4]);  /* number of neighbors */
    W       = mxGetM(prhs[3]);  /* number of words     */
    AnnIms  = mxGetNumberOfElements(prhs[2]);  /* number annotated images*/
        
    if (nlhs==2){ /*  Set the output pointer to the output matrix.*/
        plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(D, 1, mxREAL);
        
        f       = mxGetPr(plhs[0]);
        g       = mxGetPr(plhs[1]);}
    else{
        plhs[0] = mxCreateDoubleMatrix(W, I, mxREAL);        
        f       = mxGetPr(plhs[0]);}
            
    Offsets = malloc( AnnIms   * sizeof(int) );      /* compute cumulative sum of  word counts*/
    Offsets[0] = NW[0]; 
    for (i=1;i<AnnIms;i++)
        Offsets[i] = Offsets[i-1] + NW[i];
    
    TagPropDistSigmoid(LC, AN, Offsets, AW, NN, ND, A, B, f, g, D, I, W, K);
    
    free(Offsets);
}
