#include <neuralController.h>
#include <pthread.h>
#include <stdio.h>
#include <windows.h>

void* feedInput(void* arg);

pthread_t feedInputThread;
pthread_mutex_t mutex;
input_st input = (input_st){0};

void* feedInput(void* arg) {
    FILE* fptr = fopen("pt1.txt", "r");
    char input_buffer[50] = {0};
    int ret = 0;
    char* pEnd = NULL;
    while (1) {
        if (input.available == FALSE) {
            ret = fscanf(fptr, "%s", &input_buffer);
            if (ret == EOF) {
                break;
            }
            pthread_mutex_lock(&mutex);
            input.value = (double)strtof(input_buffer, &pEnd);
            input.available = TRUE;
            pthread_mutex_unlock(&mutex);
        }
    }
}

int main(int argc, const char* argv[]) {
    pthread_mutex_init(&mutex, NULL);
    long job = 0;
    char output[50] = {0};

    pthread_create(&feedInputThread, NULL, feedInput, (void*)(&job));

    for (int i = 0; i < 100;) {
        if (input.available == TRUE) {
            printf("Value: %f\n", input.value);
            i++;
            pthread_mutex_lock(&mutex);
            input.value = 0;
            input.available = FALSE;
            pthread_mutex_unlock(&mutex);
        }
    }

    pthread_join(feedInputThread, NULL);

    return 0;
}