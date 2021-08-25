#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include "definitions.h"
#include "general.h"

BOOL readFile(char* fileName, char* seq1, char* seq2, double* weights, BOOL* isMax)
{
	FILE* pF;
    char minmax[MAX_LEN_MINMAX];
	
	pF = fopen(fileName, "r");
	if (!pF)
	{
        fprintf(stderr, "failed to open the file\n");
		return False;
	}

	for(int i = 0; i < NUM_OF_WEIGHTS; i++)
		fscanf(pF, "%lf", &weights[i]);
	
	fscanf(pF, "%s", seq1);
	fscanf(pF, "%s", seq2);
	fscanf(pF, "%s", minmax);
	
	if (strcmp(minmax,"maximum") == 0)
		*isMax = True;
	else if (strcmp(minmax,"minimum") == 0)
		*isMax = False;
	else
	{
        fprintf(stderr, "unrecognized word in the file\n");
		fclose(pF);
		return False;
	}
	
	fclose(pF);
	return True;
}

/*
	free all pointers passed to the function
	last paramater needs to be NULL
 */
void freeAll(void* ptr, ...)
{
	va_list args;
	va_start(args, ptr);
	while (ptr) // until ptr != NULL
	{
		free(ptr);
		ptr = va_arg(args, void*); //get next var
	}
	va_end(args);
}

void freeMutant(Mutant* mutant)
{
	free(mutant->mutantSeq);
	free(mutant);
}