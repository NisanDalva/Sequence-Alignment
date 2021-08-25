#include <stdlib.h>
#include <stdio.h>
#include "definitions.h"

BOOL readFile(char* fileName, char* seq1, char* seq2, double* weights, BOOL* isMax);
void freeAll(void* ptr, ...);
void freeMutant(Mutant* mutant);