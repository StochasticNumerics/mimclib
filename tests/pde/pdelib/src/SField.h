#ifndef _SFIELD_
#define _SFIELD_

typedef void *SField;
typedef const void *CSField;

int SFieldCreate(SField *sf,unsigned long seed);

int SFieldBeginRuns(SField sf, const double *mod, const unsigned int *Ns, unsigned int count);
int SFieldSample(SField sf, double *goal);
int SFieldEndRuns(SField sf);

int SFieldDestroy(SField *sf);

int MLMCInitialHierarchy(unsigned int *L, unsigned int *nelem, unsigned int *Ml);

int SFieldGetDim(SField sf);
#endif
