#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_fast_rotation(int n, int *rot, int num_rot){
    int avx = 1;
    FILE *f, *f2;
    f = fopen("fast_rotation_code.c", "w+");//generated code
    f2 = fopen("rotation_matrix", "w+");//for debugging
    int *hadamard = (int *)malloc(n*n*sizeof(int));
    int *result = (int *)malloc(n*n*sizeof(int));
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
    fwrite(result, sizeof(float),n*n,f2);//for debugging
    if(avx){
        
    }
}
