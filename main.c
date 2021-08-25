#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "cudaFunctions.h"
#include "definitions.h"
#include "general.h"


int main(int argc, char *argv[])
{
	// check for correct call
	if(argc != 2)
	{
		printf("must specified a file_name\n");
		return -1;
	}

	// all variables definitions
	int rank, size;
	char* seq1;
	char* seq2;
	int lenSeq1, lenSeq2;

	int maxOffset;
	int masterStartOffset, masterEndOffset;
	int slaveStartOffset, slaveEndOffset;

	int offsetOmpStart, offsetOmpEnd;
	int offsetCudaStart, offsetCudaEnd;
	
	double weights[NUM_OF_WEIGHTS];
	BOOL isMax;
	Mutant* bestMutant;

	// -------------------------- MPI section --------------------------
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2)
	{
		fprintf(stderr, "must run with 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double start = MPI_Wtime();

	// both processes allocate memory for seq1 and seq2
	seq1 = (char*) malloc(MAX_LEN_SEQ1 * sizeof(char));
	if(!seq1)
	{
		fprintf(stderr, "malloc failed\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}
	
	seq2 = (char*) malloc(MAX_LEN_SEQ2 * sizeof(char));
	if(!seq1)
	{
		free(seq1);
		fprintf(stderr, "malloc failed\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}
	
	if (rank == 0)
	{
		// only master reads the input file
		BOOL succeed = readFile(argv[1], seq1, seq2, weights, &isMax);
		if(!succeed)
		{
			freeAll(seq1, seq2, NULL);
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}

		// send necessary data to the other process
		MPI_Send(weights, 4, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(seq1, MAX_LEN_SEQ1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		MPI_Send(seq2, MAX_LEN_SEQ2, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&isMax, 1, MPI_C_BOOL, 1, 0, MPI_COMM_WORLD);
	}
	else
	{
		// receive the data from the master process
		MPI_Recv(weights, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(seq1, MAX_LEN_SEQ1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(seq2, MAX_LEN_SEQ2, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&isMax, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD,&status);
	}

	lenSeq1 = strlen(seq1);
	lenSeq2 = strlen(seq2);
	maxOffset = lenSeq1 - lenSeq2 + 1; // allways positive because len of seq1 is greater than len of seq2

	masterStartOffset = 0;
	masterEndOffset = maxOffset / 2;
	
	slaveStartOffset = masterEndOffset;
	slaveEndOffset = maxOffset;

	// calculate start and end offsets for OpenMP and CUDA
	if(rank == 0)
	{
		offsetOmpStart = masterStartOffset;
		offsetOmpEnd = masterEndOffset / 2;

		offsetCudaStart = offsetOmpEnd;
		offsetCudaEnd = masterEndOffset;
	}
	else
	{
		offsetOmpStart = slaveStartOffset;
		offsetOmpEnd = slaveStartOffset + (maxOffset - offsetOmpStart) / 2;

		offsetCudaStart = offsetOmpEnd;
		offsetCudaEnd = maxOffset;
	}
	
	bestMutant = (Mutant*) malloc(sizeof(Mutant));
	if (!bestMutant)
	{
		fprintf(stderr, "malloc failed\n");
		freeAll(seq1, seq2, NULL);
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	// initialize the best mutant for both processes
	if(isMax)
		bestMutant->score = -INFINITY;
	else
		bestMutant->score = INFINITY;
	
	bestMutant->mutantSeq = seq2;
	bestMutant->length = lenSeq2;
	bestMutant->offset = -1;

	// -------------------------- openMP section --------------------------
	omp_set_num_threads(4);
	Mutant* currentMutant;
	
#pragma omp parallel for
	for (int i = offsetOmpStart; i < offsetOmpEnd; i++)
	{
		currentMutant = findMutantBestScoreGivenOffset(seq1, seq2, lenSeq1, lenSeq2, i, weights, isMax);
		#pragma omp critical // to update the mutant
		{
			// update the mutant if needed
			if (needToUpdateMutant(currentMutant, bestMutant, isMax))
			{
				bestMutant->score = currentMutant->score;
				bestMutant->mutantSeq = currentMutant->mutantSeq;
				bestMutant->offset = i;
			}
		}
	}
	freeMutant(currentMutant);

	// -------------------------- CUDA section --------------------------
	BOOL succeed = computeOnGPU(bestMutant, seq1, seq2, lenSeq1, lenSeq2, weights, isMax, offsetCudaStart, offsetCudaEnd);
	if (!succeed)
	{
		fprintf(stderr, "cant compute with CUDA\n");
		freeAll(seq1, seq2, NULL);
		freeMutant(bestMutant);
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}
	
	// -------------------------- back to MPI --------------------------
	if (rank == 0)
	{
		// receive all the results
		char* receivedMutantSeq = (char*) malloc(lenSeq2 * sizeof(char));
		if (!receivedMutantSeq)
		{
			fprintf(stderr, "malloc failed\n");
			freeAll(seq1, seq2, NULL);
			freeMutant(bestMutant);
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}

		int receivedOffset;
		double receivedScore;
		
		MPI_Recv(receivedMutantSeq, lenSeq2, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&receivedOffset, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&receivedScore, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&status);

		// update the mutant if needed
		if((isMax && receivedScore > bestMutant->score) ||
			(!isMax && receivedScore < bestMutant->score))
		{
			bestMutant->score = receivedScore;
			bestMutant->mutantSeq = receivedMutantSeq;
			bestMutant->mutantSeq[lenSeq2] = '\0';
			bestMutant->offset = receivedOffset;
		}

		printf("finish calculation in %.3lf seconds\n",MPI_Wtime() - start);
		
		printf("best mutant:\t%s\n", bestMutant->mutantSeq);
		printf("offset:\t%d\n", bestMutant->offset);
		printf("score:\t%.3lf\n", bestMutant->score);

		// write to file
		FILE* pF = fopen(OUTPUT_FILE_NAME, "w");
		if (!pF)
		{
			printf("failed to open the file\n");
			freeAll(seq1, seq2, NULL);
			freeMutant(bestMutant);
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}

		fprintf(pF,"%s\n", bestMutant->mutantSeq);
		fprintf(pF,"%d\n", bestMutant->offset);
		fprintf(pF,"%.3lf", bestMutant->score);
		
		freeMutant(bestMutant);
		fclose(pF);
	}
	else
	{
		// send the results to the master
		MPI_Send(bestMutant->mutantSeq, lenSeq2, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&bestMutant->offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&bestMutant->score, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	// both processes free those allocations
	freeAll(seq1, seq2, NULL);

	MPI_Finalize();
}


BOOL needToUpdateMutant(Mutant* firstMutant, Mutant* secondMutant, BOOL isMax)
{
	if ((isMax && firstMutant->score > secondMutant->score) || // looking for highest score
		(!isMax && firstMutant->score < secondMutant->score)) // looking for lowest score
		return True;
	
	return False;
}