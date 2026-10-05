/* Thread A allocates and touches 2 MB; thread B touches it again and frees it.
   Peak 2 MB; >= 2 MB freed. */
#include <pthread.h>
#include <stdlib.h>
#include "common.h"

static void *producer(void *arg)
{
    char *p = malloc(2 * MB);
    touch(p, 2 * MB);
    return p;
}

static void *consumer(void *p)
{
    touch(p, 2 * MB);
    free(p);
    return 0;
}

int main(void)
{
    pthread_t t;
    void *p;
    pthread_create(&t, 0, producer, 0);
    pthread_join(t, &p);
    pthread_create(&t, 0, consumer, p);
    pthread_join(t, 0);
    return 0;
}
