#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "svm.h"
#include "svmtools.h"

struct svm_problem *read_svm_problem1(const char *filename);

static void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}

int main(int argc, char *argv[])
{
    struct svm_problem *prob;
    struct svm_model *model;
    int i;
    double *predictY;
    FILE *outf;
    
    if(argc != 4) {
        fprintf(stderr, "\n  Usage: %s in.svm model_file out.txt\n", argv[0]);
        fprintf(stderr,"  in.svm: y-values can be omitted!!!\n");
        exit(EXIT_FAILURE);
    }
    
    outf = fopen(argv[3],"w");
    if(outf == NULL) {
        fprintf(stderr,"Error: can't open file %s\n",argv[3]);
        exit(EXIT_FAILURE);
    }

    prob = read_svm_problem1(argv[1]);
    model = svm_load_model(argv[2]);
    predictY = predict(model,prob);
    for(i=0; i<prob->l; ++i)
        fprintf(outf,"%g\n",predictY[i]);
    fclose(outf);
    
    free_svm_problem(prob);
    svm_free_and_destroy_model(&model);
    free(predictY);

    printf("\n  predicted values ared saved in `%s`\n\n",argv[3]);
    
    return 0;
}

static char* readline(FILE *input)
{
    int len;
    int max_line_len = 1024;
    char *line = Malloc(char,max_line_len);
    
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

struct svm_problem *read_svm_problem1(const char *filename)
{
    int elements, max_index, inst_max_index, i, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;
    char *line;
    char *p;
    char *tmp;
    int has_no_y;
    struct svm_problem *prob = Malloc(struct svm_problem,1);
    struct svm_node *x_space;
    int errno;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob->l = 0;
    elements = 0;

    has_no_y = 0;
    while((line = readline(fp))!=NULL)
    {
        p = strtok(line," \t"); // label

        if(strchr(p,':') != NULL) {  // has no y-value
            ++elements;
            has_no_y = 1;
        }
        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements;
        ++prob->l;
    }
    rewind(fp);

    if(has_no_y)
        prob->y = NULL;
    else
        prob->y = Malloc(double,prob->l);
    prob->x = Malloc(struct svm_node *,prob->l);
    x_space = Malloc(struct svm_node,elements);

    max_index = 0;
    j=0;
    for(i=0;i<prob->l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        line = readline(fp);
        prob->x[i] = &x_space[j];
        label = strtok(line," \t");
        if(has_no_y) { // has no y-value
            tmp = strchr(label,':');
            if(tmp != NULL) {  // some of samples have y-values, just pass those!!!!
                *tmp = '\0';
                errno = 0;
                x_space[j].index = (int)strtol(label,&endptr,10);
                if(endptr == label || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                    exit_input_error(i+1);
                else
                    inst_max_index = x_space[j].index;
                errno = 0;
                x_space[j].value = strtod(tmp+1,&endptr);
                if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                    exit_input_error(i+1);
                ++j;
            }
        }
        else { // 1st column is y-value
            prob->y[i] = strtod(label,&endptr);
            if(endptr == label)
                exit_input_error(i+1);
        }

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }

    fclose(fp);
    return prob;
}

