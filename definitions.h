#pragma once

#define MAX_LEN_SEQ1	10001
#define MAX_LEN_SEQ2	5001
#define MAX_LEN_MINMAX  7
#define NUM_OF_WEIGHTS	4

#define CONSERVATIVE_GROUPS_LEN 9
#define SEMI_CONSERVATIVE_GROUPS_LEN 11
#define ALPHABET_LEN 26

#define OUTPUT_FILE_NAME "output.txt"

typedef enum {
	False,
	True
}BOOL;

typedef struct {
	char* mutantSeq;
	int length;
	double score;
	int offset;
}Mutant;

typedef struct {
	int mutantOffset;
	char charToPut;
	int charOffset;
	double score;
}CudaResult;

BOOL needToUpdateMutant(Mutant* currentMutant, Mutant* bestMutant, BOOL isMax);