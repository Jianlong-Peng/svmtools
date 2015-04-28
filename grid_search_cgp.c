/*=============================================================================
#     FileName: grid_search.c
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2014-07-08 13:22:49
#   LastChange: 2014-07-08 13:28:49
#      History:
=============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include "svm.h"
#include "svmtools.h"

#if defined(USE_THREAD)
extern int num_thread;
#endif

void print_null(const char *s) {}

int main(int argc, char *argv[])
{
    struct svm_problem *prob;
    struct svm_parameter *para;

#if defined(USE_THREAD)
    if(argc!=2 && argc!=3) {
        fprintf(stderr, "\n  Usage: %s in.svm [num_thread]\n\n", argv[0]);
        fprintf(stderr, "  [num_thread]: specify the number of threads to be used\n");
        fprintf(stderr, "                (default: 5)\n\n");
        fprintf(stderr, "  OBJ: to search best c,g,p for epsilon-SVR using RBF kernel\n\n");
        exit(EXIT_FAILURE);
    }
    if(argc == 3)
        num_thread = atoi(argv[2]);
#else
    if(argc != 2) {
        fprintf(stderr, "\n  Usage: %s in.svm\n", argv[0]);
        fprintf(stderr, "  OBJ: to search best c,g,p for epsilon-SVR using RBF kernel\n\n");
        exit(EXIT_FAILURE);
    }
#endif

    svm_set_print_string_function(print_null);
    
    prob = read_svm_problem(argv[1]);
    para = create_svm_parameter(EPSILON_SVR, RBF);
    grid_search_cpg(prob, para, 5, 1, calcR2);
    free_svm_problem(prob);
    svm_destroy_param(para);

    return 0;
}
