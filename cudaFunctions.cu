#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "definitions.h"
#include "cudaFunctions.h"

/*
	find the best mutant, the best one will return by pointer
	return True if succeed, otherwise - False
*/
__host__ BOOL computeOnGPU(Mutant* bestMutant, char* seq1, char* seq2, int lenSeq1, int lenSeq2,
				 double weights[], BOOL isMax, int startOffset,int endOffset)
{		
	int numOfOffsetsToCalc = endOffset - startOffset + 1;

	if(numOfOffsetsToCalc == 0) // if is it equal to 0, no need to do somthing
		return True;

	// error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	
	// determine the size of memory to allocate
	int sizeMemSeq1 = lenSeq1 * sizeof(char);
    int sizeMemSeq2 = lenSeq2 * sizeof(char);
	int sizeWeights = 4 * sizeof(double);
 	int sizeResults = numOfOffsetsToCalc * sizeof(CudaResult);
    
	// allocate on GPU
	char* gpuSeq1;
	char* gpuSeq2;
	double* gpuWeights;
	CudaResult* gpuResults;

	err = cudaMalloc((void**) &gpuSeq1, sizeMemSeq1);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return False;
	}
	
	err = cudaMalloc((void**) &gpuSeq2, sizeMemSeq2);
	if (err != cudaSuccess)
	{
		cudaFree(gpuSeq1);
		fprintf(stderr, "failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return False;
	}

	err = cudaMalloc((void**) &gpuWeights, sizeWeights);
	if (err != cudaSuccess)
	{
		cudaFreeAll(gpuSeq1, gpuSeq2, NULL);
		fprintf(stderr, "failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return False;
	}

	err = cudaMalloc((void**) &gpuResults, sizeResults);
	if (err != cudaSuccess)
	{
		cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, NULL);
		fprintf(stderr, "failed to allocate device memory - %s\n", cudaGetErrorString(err));
        return False;
	}

	// copy data from host to the GPU memory	
	if(cudaMemcpy(gpuSeq1, seq1, sizeMemSeq1, cudaMemcpyHostToDevice) != cudaSuccess
		|| cudaMemcpy(gpuSeq2, seq2, sizeMemSeq2, cudaMemcpyHostToDevice) != cudaSuccess
		|| cudaMemcpy(gpuWeights, weights, sizeWeights, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, gpuResults, NULL);
    	fprintf(stderr, "failed to copy data from host to device \n");
     	return False;
	}
	
   	// launch the Kernel
   	int threadsPerBlock = calcNumThreadsPerBlock(numOfOffsetsToCalc);
    int blocksPerGrid = (numOfOffsetsToCalc + threadsPerBlock - 1) / threadsPerBlock;

    findBestMutant<<<blocksPerGrid, threadsPerBlock>>>(gpuResults, gpuSeq1, gpuSeq2, lenSeq1, lenSeq2, gpuWeights,
														isMax, startOffset, endOffset);
   	
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
		cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, gpuResults, NULL);
    	fprintf(stderr, "failed to launch the kernel - %s\n", cudaGetErrorString(err));
    	return False;
    }
	
	CudaResult* cpuResults = (CudaResult*) malloc(sizeResults);
	if(!cpuResults)
	{
		cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, gpuResults, NULL);
		fprintf(stderr, "malloc failed\n");
    	return False;
	}

	err = cudaMemcpy(cpuResults, gpuResults,sizeResults, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
   	{
		cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, gpuResults, cpuResults, NULL);
    	fprintf(stderr, "failed to copy result from device to host - %s\n", cudaGetErrorString(err));
    	return False;
    }

	for(int i = 0; i < numOfOffsetsToCalc; i ++) 
	{
		// check if better result found
		if ((isMax && cpuResults[i].score > bestMutant->score) || // looking for highest score
			(!isMax && cpuResults[i].score < bestMutant->score)) // looking for lowest score
		{
			bestMutant->score = cpuResults[i].score;
			bestMutant->offset = cpuResults[i].mutantOffset;
			
			// check if substitute char is needed
			if(cpuResults[i].charToPut != '-')
				substituteChar(bestMutant, cpuResults, i, seq2, lenSeq2);
		}
	}
	
	cudaFreeAll(gpuSeq1, gpuSeq2, gpuWeights, gpuResults, NULL);
	return True;
}

__host__ int calcNumThreadsPerBlock(int numOfOffsetsToCalc)
{
	// calculate threads needed and find power of 2 that fits the job
	int numThreadsNeeded = 1;
	while (numThreadsNeeded < numOfOffsetsToCalc)
		numThreadsNeeded *= 2;
	

	int threadsInBlock = 1;
	int i = threadsInBlock * 2;

	while(i < 1024 + 1)
	{
		if(numThreadsNeeded % numOfOffsetsToCalc > numThreadsNeeded % i)
			threadsInBlock = i;
		i *= 2;
	}
	return threadsInBlock;
}

/*
	substitute a character based on given results
	this functuion invoked only after we checked if we need to substitute at all
*/
__host__ void substituteChar(Mutant* mutant, CudaResult* result, int i, char* seq, int lenSeq)
{
		memcpy(mutant->mutantSeq, seq, lenSeq * sizeof(char));
		mutant->mutantSeq[lenSeq+1] = '\0';
		mutant->mutantSeq[result[i].charOffset] = result[i].charToPut;
}


__global__  void findBestMutant(CudaResult* result, char* seq1, char* seq2, int lenSeq1, int lenSeq2,
				 double weights[], BOOL isMax, int startOffset,int endOffset)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = tId + startOffset;
	
	if(offset < startOffset || offset > endOffset)
		return;

	Mutant* foundedMutant = findMutantBestScoreGivenOffset(seq1, seq2, lenSeq1, lenSeq2, offset, weights, isMax);
	if(!foundedMutant)
		return;

	// initialization
	result[tId].charToPut = '-';
	result[tId].charOffset = -1;

	for(int i = 0; i < lenSeq2; i++)
	{
		if (foundedMutant->mutantSeq[i] != seq2[i])
		{
			result[tId].charToPut = foundedMutant->mutantSeq[i];
			result[tId].charOffset = i;
		}
	}
	result[tId].mutantOffset = offset;
	result[tId].score = foundedMutant->score;

	cudaFreeMutant(foundedMutant);
}

/*
	given an offset, search for a best mutant as possible
	returns a pointer to a Mutant
*/
__host__ __device__ Mutant* findMutantBestScoreGivenOffset(char* seq1, char* seq2, int lenSeq1, int lenSeq2,
															int offset, double weights[], BOOL isMax)
{
	char* resultsSeq;
	double effect = 0;
	double checkedEffect = 0;
	Mutant* bestMutant = (Mutant*) malloc(sizeof(Mutant));
	if (!bestMutant)
	{
		printf("malloc failed\n");
		return NULL;
	}
	
	// initialize
	bestMutant->mutantSeq = (char*)malloc(lenSeq2 * sizeof(char) + 1);
	if (!bestMutant->mutantSeq)
	{
		printf("malloc failed\n");
		return NULL;
	}
	memcpy(bestMutant->mutantSeq, seq2, lenSeq2 * sizeof(char));
	bestMutant->length = lenSeq2;
	resultsSeq = getResultsSequence(seq1, seq2, lenSeq1, lenSeq2, offset);
	
	double startScore = calcScore(resultsSeq,lenSeq2,weights);
	bestMutant->score  = startScore;

	const char alphabet[ALPHABET_LEN] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
		'P','Q','R','S','T','U','V','W','X','Y','Z'};

	// run all over seq2
	for (int i = 0; i < lenSeq2; i++)
	{
		// run all over the alphabet
		for (int j = 0; j < ALPHABET_LEN; j++)
		{
			// if both are not in conservative groups, then we can substitute a character
			if (!inConservativeGroups(alphabet[j], seq2[i]))
			{
				checkedEffect = calcChangeBetweenTwoChars(seq1[i + offset], seq2[i], alphabet[j], weights);
				
				// check if we found a better one
				if ((isMax && checkedEffect > effect) || // looking for highest score
				(!isMax && checkedEffect < effect) ) // looking for lowest score
				{
					bestMutant->mutantSeq[i] = alphabet[j]; // substitute
					effect = checkedEffect;
				}
			}
		}
	}

	bestMutant->score = startScore + effect;
	return bestMutant;
}



__host__ __device__ char getSign(char c1, char c2)
{
	if (c1 == c2)
		return '*';
	
	if (inConservativeGroups(c1, c2))
		return ':';
	
	if (inSemiConservativeGroups(c1, c2))
		return '.';
	
	if (c2 == '-')
		return '-';
	
	return ' ';
}


__host__ __device__ char* getResultsSequence(char* seq1, char* seq2 , int size1, int size2, int offset)
{
	char* results = (char*)malloc(size2 * sizeof(char) + 1);
	if(!results)
	{
		printf("malloc failed\n");
		return NULL;
	}
	
	for (int i = 0; i < size2; i++)
		results[i] = getSign(seq1[i + offset], seq2[i]);

	return results;
}


__host__ __device__ double calcScore(char* results, int resultsLen, double* weights) 
{
	int stars = 0;
	int colons = 0;
	int points = 0;
	int spaces = 0;
	double totalScore = 0;

	for (int i = 0; i < resultsLen + 1; i++) 
	{
		if (results[i] == '*')
			stars++;
		else if (results[i] == ':')
			colons++;
		else if (results[i] == '.')
			points++;
		else if (results[i] == ' ')
			spaces++;
	}

	totalScore = weights[0] * stars - weights[1] * colons - weights[2] * points - weights[3] * spaces;
	return totalScore;
}

/*
	calculate how much one letter from the alphabet affects the result
	in relation to the letter from seq2
*/
__host__ __device__ double calcChangeBetweenTwoChars(char seq1Char, char seq2Char, char anotherChar, double weights[])
{
	if(anotherChar == '-')
		return 0;
	
	double currentScore;
	if(seq2Char == seq1Char) // star
		currentScore = weights[0];

	else if(inConservativeGroups(seq2Char,seq1Char)) // colon
		currentScore = -weights[1];

	else if(inSemiConservativeGroups(seq2Char,seq1Char)) // point
		currentScore = -weights[2];

	else // space
		currentScore = -weights[3];
	
	
	double checkedScore;
	if(anotherChar == seq1Char)
		checkedScore = weights[0];

	else if(inConservativeGroups(anotherChar,seq1Char))
		checkedScore = -weights[1];

	else if(inSemiConservativeGroups(anotherChar,seq1Char))
		checkedScore = -weights[2];

	else
		checkedScore = -weights[3];
	
	return checkedScore - currentScore;
}

__host__ __device__ BOOL inConservativeGroups(char c1, char c2)
{
	const char* ConservativeGroups[CONSERVATIVE_GROUPS_LEN] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK",
		"FYW", "HY", "MILF"};

	for (int i = 0; i < CONSERVATIVE_GROUPS_LEN; i++)
		if (isExistInSeq(ConservativeGroups[i], c1, c2))
			return True;

	return False;

}

__host__ __device__ BOOL inSemiConservativeGroups(char c1, char c2)
{
	const char* SemiConservativeGroups[SEMI_CONSERVATIVE_GROUPS_LEN] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK",
		"NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

	for (int i = 0; i < SEMI_CONSERVATIVE_GROUPS_LEN; i++)
		if (isExistInSeq(SemiConservativeGroups[i], c1, c2))
			return True;

	return False;
}

__host__ __device__ BOOL isExistInSeq(const char* seq, char c1, char c2)
{
	if (strChar(seq, c1) && strChar(seq, c2))
		return True;
	return False;
}


__host__ __device__ int strLength(const char *str)
{
    int length = 0;
    while (*str++)
        length++;

    return length;
}

__host__ __device__ int strCompare(const char* s1, const char* s2)
{
    while(*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }

    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}


__host__ __device__ const char* strChar(const char* s, const char c)
{
	while(*s != c && *s != '\0')
		s++;
	
	if (*s == c)
		return s;
	
	return NULL;
}

__host__ void cudaFreeAll(void *ptr, ...)
{
	va_list args;
	va_start(args, ptr);
	while (ptr) // until ptr != NULL
	{
		cudaError_t err = cudaFree(ptr);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "failed to free device data\n");
    		return;
		}
		ptr = va_arg(args, void*); //get next var
	}
	va_end(args);
}

__device__ void cudaFreeMutant(Mutant* mutant)
{
	free(mutant->mutantSeq);
	free(mutant);
}