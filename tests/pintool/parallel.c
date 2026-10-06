/* 8 threads run concurrently: each allocates and touches a private 1 MB block
   while all of them touch disjoint slices of a shared 4 MB array; after a
   barrier each frees its private block. Peak 12 MB; >= 12 MB freed. */
#include <pthread.h>
#include <stdlib.h>
#include "common.h"

#define THREADS 8
static char *shared;
static pthread_barrier_t barrier;

static void *worker(void *arg)
{
    long id = (long)arg;
    char *private = malloc(1 * MB);
    touch(private, 1 * MB);
    touch(shared + id * (4 * MB / THREADS), 4 * MB / THREADS);
    pthread_barrier_wait(&barrier); /* everything is live at once here */
    free(private);
    return 0;
}

int main(void)
{
    pthread_t t[THREADS];
    shared = malloc(4 * MB);
    pthread_barrier_init(&barrier, 0, THREADS);
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], 0, worker, (void *)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], 0);
    free(shared);
    return 0;
}
