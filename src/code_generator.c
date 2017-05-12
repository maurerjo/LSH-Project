#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_fast_rotation(int n, int *rot, int num_rot, char *name){
    int avx = 1;
    FILE *f, *f2;
    f = fopen("fast_rotation_code.c", "w+");//generated code
    f2 = fopen("rotation_matrix", "w+");//for debugging
    int *hadamard = (int *)malloc(n*n*sizeof(int));
    float *result = (float *)malloc(n*n*sizeof(float));
    for(int i = 0;i<n;i++)
        for(int ii = 0; ii<n;ii++){
            hadamard[i*n+ii] = pow(-1,__builtin_popcount(i&ii));
            result[i*n+ii] = pow(-1,__builtin_popcount(i&ii));
        }

    for(int i = 0;i<n;i++){
        for(int ii = 0;ii<n;ii++){
            result[i*n+ii]*=rot[i];
        }
    }
    for(int r = 1;r<num_rot;r++){
        for(int i = 0;i<n;i++){
            for(int ii = 0;ii<n;ii++){
                for(int i3 = 0;i3<n;i3++){
                    result[i*n+ii] += result[i*n+i3]*hadamard[ii*n+i3];//Hadamard is it's own transpose
                }
            }
        }
        for(int i = 0;i<n;i++){
            for(int ii = 0;ii<n;ii++){
                result[i*n+ii]*=rot[r*n+i];//or +ii
            }
        }
    }
    float scalar = pow(2,-num_rot*log2(n)/2.0);
    for(int i = 0;i<n;i++){
        for(int ii = 0;ii<n;ii++){
            result[i*n+ii]*=scalar;//rescale to be distance preserving
        }
    }
    fwrite(result, sizeof(float),n*n,f2);//for debugging
    if(avx){

    }else{
        for(int i = 0; i < n;i++){
            fwrite("x["+i+"]=0",sizeof(char),10,f);
            for(int ii = 0; ii < n;ii++){
                if(abs(result[i][ii])==0){
                    //ignore term
                }
                else if (abs(result[i][ii])==1){
                    fwrite("+x["+ii+"]",sizeof(char),10,f);//save multiply
                }else{
                    fwrite("+x["+ii+"]*"+result[i][ii],sizeof(char),20,f);
                }
            }
            fwrite(";/n",sizeof(char),3,f);
        }
    }
}
