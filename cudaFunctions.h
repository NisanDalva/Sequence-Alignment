#pragma once
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include "definitions.h"

__host__ BOOL computeOnGPU(Mutant* bestMutant, char* seq1, char* seq2, int lenSeq1, int lenSeq2,
				 double weights[], BOOL isMax, int startOffset,int endOffset);

__host__ int calcNumThreadsPerBlock(int numOfOffsetsToCalc);
__host__ void substituteChar(Mutant* mutant, CudaResult* result, int i, char* seq, int lenSeq);

__global__  void findBestMutant(CudaResult* result, char* seq1, char* seq2, int lenSeq1, int lenSeq2,
				 double weights[], BOOL isMax, int startOffset,int endOffset);

__host__ __device__ Mutant* findMutantBestScoreGivenOffset(char* seq1, char* seq2, int lenSeq1, int lenSeq2,
															int offset, double weights[], BOOL isMax);


__host__ __device__ char getSign(char c1, char c2);
__host__ __device__ char* getResultsSequence(char* seq1, char* seq2 , int size1, int size2, int offset);
__host__ __device__ double calcScore(char* results, int resultsLen, double* weights) ;

__host__ __device__ double calcChangeBetweenTwoChars(char seq1Char, char seq2Char, char anotherChar, double weights[]);

__host__ __device__ BOOL inConservativeGroups(char c1, char c2);
__host__ __device__ BOOL inSemiConservativeGroups(char c1, char c2);
__host__ __device__ BOOL isExistInSeq(const char* seq, char c1, char c2);

__host__ __device__ int strLength(const char *str);
__host__ __device__ int strCompare(const char* s1, const char* s2);
__host__ __device__ const char* strChar(const char* s, const char c);

__host__ void cudaFreeAll(void *ptr, ...);
__device__ void cudaFreeMutant(Mutant* mutant);