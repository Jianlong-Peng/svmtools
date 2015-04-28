/*=============================================================================
#     FileName: svmtools.c
#         Desc: 
#       Author: jlpeng
#        Email: jlpeng1201@gmail.com
#     HomePage: 
#      Created: 2013-06-06 17:00:13
#   LastChange: 2014-07-08 14:03:37
#      History:
=============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <errno.h>
#include <float.h>
#include "svm.h"
#include "svmtools.h"

//#define USE_THREAD

static void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
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

struct svm_problem *read_svm_problem(const char *filename)
{
    int elements, max_index, inst_max_index, i, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;
    char *line;
    char *p;
    struct svm_problem *prob = Malloc(struct svm_problem,1);
    struct svm_node *x_space;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob->l = 0;
    elements = 0;

    while((line = readline(fp))!=NULL)
    {
        p = strtok(line," \t"); // label

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
        prob->y[i] = strtod(label,&endptr);
        if(endptr == label)
            exit_input_error(i+1);

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

struct svm_problem *copy_part_svm_problem(const struct svm_problem *prob, const int *mask)
{
    int i,j,n;
    struct svm_problem *newprob = Malloc(struct svm_problem, 1);
    int num_features = svm_num_features(prob);
    int *newindex = Malloc(int, num_features);
    i = 1;
    for(j=0; j<num_features; ++j) {
        if(mask[j] == 1) {
            newindex[j] = i;
            ++i;
        }
        else
            newindex[j] = 0;
    }
    newprob->l = prob->l;
    newprob->y = Malloc(double, newprob->l);
    newprob->x = Malloc(struct svm_node*, newprob->l);
    for(i=0; i<newprob->l; ++i) {
        newprob->y[i] = prob->y[i];
        j = 0;
        n = 0;
        while(prob->x[i][j].index != -1) {
            if(mask[prob->x[i][j].index-1] == 1)
                ++n;
            ++j;
        }
        newprob->x[i] = Malloc(struct svm_node, n+1);
        j = 0;
        n = 0;
        while(prob->x[i][j].index != -1) {
            if(mask[prob->x[i][j].index-1] == 1) {
                newprob->x[i][n].index = newindex[prob->x[i][j].index-1];
                newprob->x[i][n].value = prob->x[i][j].value;
                ++n;
            }
            ++j;
        }
        newprob->x[i][n].index = -1;
    }
    free(newindex);
    return newprob;
}

struct svm_problem *copy_svm_problem(const struct svm_problem *prob)
{
    struct svm_problem *newprob;
    int num_features = svm_num_features(prob);
    int *mask = Malloc(int, num_features);
    int i;

    for(i=0; i<num_features; ++i)
        mask[i] = 1;
    
    newprob = copy_part_svm_problem(prob,mask);
    free(mask);
    return newprob;
}

static void print_problem(const struct svm_problem *prob, FILE *outf)
{
    int i,j;
    for(i=0; i<prob->l; ++i) {
        if(prob->y != NULL)
            fprintf(outf,"%g",prob->y[i]);
        j = 0;
        while(prob->x[i][j].index != -1) {
            fprintf(outf, " %d:%g", prob->x[i][j].index, prob->x[i][j].value);
            ++j;
        }
        fprintf(outf,"\n");
    }
}

void write_svm_problem(const struct svm_problem *prob, const char *outfile)
{
    FILE *outf;

    if(prob == NULL)
        return;
    
    outf = fopen(outfile,"w");
    if(outf == NULL) {
        fprintf(stderr,"Error: failed to open file %s\n",outfile);
        exit(EXIT_FAILURE);
    }
    print_problem(prob,outf);
    fclose(outf);
}

void display_svm_problem(const struct svm_problem *prob)
{
    print_problem(prob,stdout);
}

void free_svm_problem(struct svm_problem *prob)
{
    if(prob == NULL)
        return;
    if(prob->y)
        free(prob->y);
    if(prob->x) {
        free(prob->x[0]);
        free(prob->x);
    }
    free(prob);
}

void free_svm_problem1(struct svm_problem *prob)
{
    int i;
    if(prob->y)
        free(prob->y);
    if(prob->x) {
        for(i=0; i<prob->l; ++i)
            free(prob->x[i]);
        free(prob->x);
    }
    free(prob);
}

struct svm_parameter* create_svm_parameter(int svm_type, int kernel_type)
{
    struct svm_parameter *para = Malloc(struct svm_parameter, 1);
    para->svm_type = svm_type;
    para->kernel_type = kernel_type;
    para->degree = 3;
    para->gamma = 0.;
    para->coef0 = 0;
    para->cache_size = 100;
    para->eps = 0.001;
    para->C = 1.0;
    para->nr_weight = 0;
    para->weight_label = NULL;
    para->weight = NULL;
    para->nu = 0.5;
    para->p = 0.1;
    para->shrinking = 1;
    para->probability = 0;
    return para;
}

struct svm_parameter* copy_svm_parameter(const struct svm_parameter *param)
{
    int i;
    struct svm_parameter *newparam = Malloc(struct svm_parameter, 1);
    newparam->svm_type = param->svm_type;
    newparam->kernel_type = param->kernel_type;
    newparam->degree = param->degree;
    newparam->gamma = param->gamma;
    newparam->coef0 = param->coef0;
    newparam->cache_size = param->cache_size;
    newparam->eps = param->eps;
    newparam->C = param->C;
    newparam->nr_weight = param->nr_weight;
    if(newparam->nr_weight != 0) {
        newparam->weight_label = Malloc(int, newparam->nr_weight);
        newparam->weight = Malloc(double, newparam->nr_weight);
        for(i=0; i<newparam->nr_weight; ++i) {
            newparam->weight_label[i] = param->weight_label[i];
            newparam->weight[i] = param->weight[i];
        }
    }
    else {
        newparam->weight_label = NULL;
        newparam->weight = NULL;
    }
    newparam->nu = param->nu;
    newparam->p = param->p;
    newparam->shrinking = param->shrinking;
    newparam->probability = param->probability;
    
    return newparam;
}


// x' * y
static double linear_kernel(const struct svm_node *x, const struct svm_node *y, int ignore_tag)
{
    double value = 0.;
    int i=0, j=0;
    while(x[i].index!=-1 && y[j].index!=-1) {
        if(x[i].index==ignore_tag)
            ++i;
        else if(y[j].index == ignore_tag)
            ++j;
        else {
            if(x[i].index < y[j].index)
                ++i;
            else if(x[i].index > y[j].index)
                ++j;
            else {
                value += ((x[i].value) * (y[i].value));
                ++i;
                ++j;
            }
        }
    }
    return value;
}
// (gamma*x'*y + coef0)^degree
static double poly_kernel(const struct svm_node *x, const struct svm_node *y, const struct svm_parameter *param, int ignore_tag)
{
    double value = linear_kernel(x,y,ignore_tag);
    return pow((param->gamma) * value + (param->coef0), param->degree);
}
// exp(-gamma*|x-y|^2)
static double rbf_kernel(const struct svm_node *x, const struct svm_node *y, const struct svm_parameter *param, int ignore_tag)
{
    double value = 0.;
    int i=0, j=0;
    
    while(x[i].index!=-1 && y[j].index!=-1) {
        if(x[i].index == ignore_tag)
            ++i;
        else if(y[j].index == ignore_tag)
            ++j;
        else {
            if(x[i].index < y[j].index) {
                value += pow(x[i].value,2);
                ++i;
            }
            else if(x[i].index > y[i].index) {
                value += pow(y[j].value,2);
                ++j;
            }
            else {
                value += pow(x[i].value-y[j].value,2);
                ++i;
                ++j;
            }
        }
    }
    while(x[i].index != -1) {
        if(x[i].index != ignore_tag)
            value += pow(x[i].value,2);
        ++i;
    }
    while(y[j].index != -1) {
        if(y[j].index != ignore_tag)
            value += pow(y[j].value,2);
        ++j;
    }
    
    return exp(-1. * (param->gamma) * value);
}
static double sigmoid_kernel(const struct svm_node *x, const struct svm_node *y, const struct svm_parameter *param, int ignore_tag)
{
    fprintf(stderr,"Error<svmtools.c::sigmodi_kernel>: Sorry, but not implemented!\n");
    exit(EXIT_FAILURE);
}

double kernel(const struct svm_node *x, const struct svm_node *y, const struct svm_parameter *param, int ignore_tag)
{
    int kernel_type = param->kernel_type;
    if(kernel_type == LINEAR)
        return linear_kernel(x,y,ignore_tag);
    else if(kernel_type == POLY)
        return poly_kernel(x,y,param,ignore_tag);
    else if(kernel_type == RBF)
        return rbf_kernel(x,y,param,ignore_tag);
    else if(kernel_type == SIGMOID)
        return sigmoid_kernel(x,y,param,ignore_tag);
    else {
        fprintf(stderr,"Error<svmtools.c::kernel>: invalid kernel type!!\n");
        exit(EXIT_FAILURE);
    }
}

int svm_num_features(const struct svm_problem *prob)
{
    int max_index=0,i,j;
    if(prob==NULL || prob->l==0)
        return 0;
    for(i=0; i<prob->l; ++i) {
        if(prob->x[i] == NULL)
            continue;
        j = 0;
        while(prob->x[i][j].index != -1) {
            if(prob->x[i][j].index > max_index)
                max_index = prob->x[i][j].index;
            ++j;
        }
    }
    return max_index;
}

// shallow copy!
static void copy_node(struct svm_node *dest, const struct svm_node *src)
{
    int i=0;
    while(src[i].index != -1)
        ++i;
    dest = Malloc(struct svm_node,i+1);
    i = 0;
    while(src[i].index != -1) {
        dest[i].index = src[i].index;
        dest[i].value = src[i].value;
        ++i;
    }
    dest[i].index = -1;
}

// train, test: should be generated by Malloc(struct svm_problem,1)
// itrains[i]: 1 - train sample; 0 - test sample
void split_svm_problem(const struct svm_problem *prob, struct svm_problem *train, 
        struct svm_problem *test, int *itrain)
{
    int i,i_train,i_test;
    train->l = 0;
    test->l = 0;
    for(i=0; i<prob->l; ++i) {
        if(itrain[i] == 1)
            ++(train->l);
        else
            ++(test->l);
    }
    train->y = Malloc(double,train->l);
    train->x = Malloc(struct svm_node*, train->l);
    test->y = Malloc(double,test->l);
    test->x = Malloc(struct svm_node*, test->l);

    i_train = 0;
    i_test = 0;
    for(i=0; i<prob->l; ++i) {
        if(itrain[i] == 1) {
            copy_node(train->x[i_train],prob->x[i]);
            train->y[i_train] = prob->y[i];
            ++i_train;
        }
        else {
            copy_node(test->x[i_test],prob->x[i]);
            test->y[i_test] = prob->y[i];
            ++i_test;
        }
    }
}

static void random_range(int total, int n, int *index)
{
    int *temp = Malloc(int,total);
    int i,j;
    int tmp;
    for(i=0; i<total; ++i)
        temp[i] = i;
    srand(time(NULL));
    for(i=0; i<total; ++i) {
        j = rand() % total;
        if(j != i) {
            tmp = temp[i];
            temp[i] = temp[j];
            temp[j] = tmp;
        }
    }
    for(i=0; i<n; ++i)
        index[i] = temp[i];
    free(temp);
}

void random_split_problem(const struct svm_problem *prob, struct svm_problem *train, 
        struct svm_problem *test, int n)
{
    int *index = Malloc(int,prob->l);
    int *remain_index = Malloc(int,n);
    int i;
    for(i=0; i<prob->l; ++i)
        index[i] = 0;
    random_range(prob->l,n,remain_index);
    for(i=0; i<n; ++i)
        index[remain_index[i]] = 1;
    split_svm_problem(prob,train,test,index);
    free(index);
    free(remain_index);
}


// train->x[i] <-> testX
// train->y[i] <-> testY
static void exchangeTrainAndTest(struct svm_problem *train, 
        struct svm_node **testX, double *testY, int i)
{
    double tempY = train->y[i];
    struct svm_node *tempX = train->x[i];
    train->x[i] = *testX;
    *testX = tempX;
    train->y[i] = *testY;
    *testY = tempY;
}

// do LOO, and evaluation method 'eval' will be called on the predicted results
double svm_loo(const struct svm_parameter *para, const struct svm_problem *prob,
        double (*eval)(const double *act, const double *pred, int n))
{
    int i,j,k;
    double testY,value;
    double *predY = Malloc(double,prob->l);
    struct svm_problem *train = Malloc(struct svm_problem, 1);
    struct svm_node *testX;
    struct svm_model *model;

    train->l = prob->l-1;
    train->y = Malloc(double,train->l);
    train->x = Malloc(struct svm_node*,train->l);

    for(i=1; i<prob->l; ++i) {
        train->y[i-1] = prob->y[i];
        k = 0;
        j = 0;
        while(prob->x[i][j].index != -1) {
            ++k;
            ++j;
        }
        ++k;
        train->x[i-1] = Malloc(struct svm_node,k);
        for(j=0; j<k; ++j) {
            train->x[i-1][j].index = prob->x[i][j].index;
            train->x[i-1][j].value = prob->x[i][j].value;
        }
    }
    testY = prob->y[0];
    k = 0;
    j = 0;
    while(prob->x[0][j].index != -1) {
        ++k;
        ++j;
    }
    ++k;
    testX = Malloc(struct svm_node,k);
    for(j=0; j<k; ++j) {
        testX[j].index = prob->x[0][j].index;
        testX[j].value = prob->x[0][j].value;
    }

    for(i=0; i<prob->l; ++i) {
        if(i > 0) {
            if(i > 1)
                exchangeTrainAndTest(train,&testX,&testY,i-2); // roll back!
            exchangeTrainAndTest(train,&testX,&testY,i-1);
        }
#if 0
        printf("training set\n");
        for(j=0; j<train->l; ++j) {
            printf("%g",train->y[j]);
            k = 0;
            while(train->x[j][k].index != -1) {
                printf(" %d:%g",train->x[j][k].index,train->x[j][k].value);
                ++k;
            }
            printf("\n");
        }
        printf("test set\n%g",testY);
        k = 0;
        while(testX[k].index != -1) {
            printf(" %d:%g",testX[k].index,testX[k].value);
            ++k;
        }
        printf("\n\n");
#endif
        model = svm_train(train,para);
        predY[i] = svm_predict(model,testX);
        svm_free_model_content(model);
    }

    // estimate
    value = eval(prob->y,predY,prob->l);

    free(predY);
    free(train->y);
    for(j=0; j<train->l; ++j)
        free(train->x[j]);
    free(train->x);
    free(train);
    free(testX);

    return value;
}


// predicted results will be saved in 'result', which is returned by 'Malloc(double,prob->l)'
double *predict(const struct svm_model *model, const struct svm_problem *prob)
{
    int i;
    double *result = Malloc(double,prob->l);
    for(i=0; i<prob->l; ++i)
        result[i] = svm_predict(model,prob->x[i]);
    return result;
}

double svm_nCV(const struct svm_parameter *para, const struct svm_problem *prob, int nfold,
        double (*eval)(const double *act, const double *pred, int n))
{
    int i, j, begin, end;
    int total = prob->l;
    double value;
    int *fold_start = Malloc(int,nfold+1);
    double *predictY = Malloc(double, total);
    struct svm_problem *train = Malloc(struct svm_problem, 1);
    struct svm_model *model;

    train->y = Malloc(double, total);
    train->x = Malloc(struct svm_node *, total);

    for(i=0; i<=nfold; ++i)
        fold_start[i] = i*total/nfold;
    for(i=0; i<nfold; ++i) {
        begin = fold_start[i];
        end = fold_start[i+1];
        train->l = 0;
        for(j=0; j<begin; ++j) { // train
            train->y[train->l] = prob->y[j];
            train->x[train->l] = prob->x[j];
            train->l++;
        }
        for(j=end; j<total; ++j) { // train
            train->y[train->l] = prob->y[j];
            train->x[train->l] = prob->x[j];
            train->l++;
        }
        //printf("\ntraining set of %dth-fold:\n",i);
        //display_svm_problem(train);
        // train and predict
        model = svm_train(train,para);
        for(j=begin; j<end; ++j)
            predictY[j] = svm_predict(model,prob->x[j]);
        svm_free_and_destroy_model(&model);
    }
    value = eval(prob->y, predictY, prob->l);
    free(fold_start);
    free(predictY);
    free(train->y);
    free(train->x);
    free(train);
    return value;
}

double calcQ2(const double *act, const double *pred, int n)
{
    int i;
    double fenzi=0., fenmu=0.;
    double act_mean=0.;
    for(i=0; i<n; ++i) {
        fenzi += pow(act[i]-pred[i],2);
        act_mean += act[i];
    }
    act_mean /= n;
    for(i=0; i<n; ++i)
        fenmu += pow(act[i]-act_mean,2);
    return 1.0-fenzi/fenmu;
}

double calcR2(const double *act, const double *pred, int n)
{
    double sum_xy=0., sum_xx=0., sum_yy=0., mean_x=0., mean_y=0.;
    double fenzi, fenmu2;
    int i;
    for(i=0; i<n; ++i) {
        sum_xy += (act[i]*pred[i]);
        sum_xx += pow(act[i],2);
        sum_yy += pow(pred[i],2);
        mean_x += act[i];
        mean_y += pred[i];
    }
    mean_x /= n;
    mean_y /= n;
    fenzi = sum_xy - n * mean_x * mean_y;
    fenmu2 = (sum_xx - n * pow(mean_x,2)) * (sum_yy - n * pow(mean_y,2));
    return pow(fenzi,2)/fenmu2;
}

double calcMAE(const double *act, const double *pred, int n)
{
    double mae=0.;
    int i;
    for(i=0; i<n; ++i)
        mae += fabs(act[i]-pred[i]);
    mae /= n;
    return mae;
}

static const double BASE_C = 2.;
static const double BASE_G = 2.;
static const double BASE_P = 0.05;

double grid_search_cp(const struct svm_problem *prob, struct svm_parameter *para,
        int nfold, int verbose, double (*eval)(const double *act, const double *pred, int n))
{
    double best_eval = DBL_MIN;
    double curr_eval;
    double best_c=para->C, best_p=para->p;
    int i,j;
    //double *target = Malloc(double, prob->l);
    for(i=-8; i<=8; ++i) {
        para->C = pow(BASE_C,i);
        for(j=-8; j<=8; ++j) {
            para->gamma = pow(BASE_G,j);
            //svm_cross_validation(prob,para,nfold,target);
            //curr_eval = eval(prob->y, target, prob->l);
            curr_eval = svm_nCV(para,prob,nfold,eval);
            if(verbose)
                printf("check c=%g, p=%g => rate=%g\n",para->C, para->p, curr_eval);
            if(curr_eval > best_eval) {
                best_c = para->C;
                best_p = para->p;
                best_eval = curr_eval;
            }
        }
    }
    para->C = best_c;
    para->p = best_p;
    if(verbose)
        printf("==> c=%g, p=%g, rate=%g\n", para->C, para->p, best_eval);
    //free(target);
    return best_eval;
}

#if defined(USE_THREAD)
struct CPG {
    int c;
    int p;
    int g;
};

struct Queue {
    int _size;
    int _capacity;
    int _head;
    int _tail;
    struct CPG *_data;
};
typedef struct Queue * QueuePtr;

QueuePtr Queue_new(int capacity)
{
    QueuePtr q = (QueuePtr)malloc(sizeof(struct Queue));
    q->_capacity = capacity;
    q->_size = 0;
    q->_head = q->_tail = -1;
    q->_data = (struct CPG*)malloc(sizeof(struct CPG)*capacity);
    return q;
}

#define Queue_free(q) (q==NULL || (free(q->_data), free(q), q=NULL))
#define Queue_empty(q) (q->_size?0:1)
#define Queue_full(q) ((q->_size==q->_capacity)?1:0)
#define Queue_head(q) (Queue_empty(q)?NULL:&(q->_data[q->_head]))

void Queue_resize(QueuePtr q, int n)
{
    int i;
    int delta = n - q->_capacity;
    if(delta > 0) {
        q->_data = (struct CPG*)realloc(q->_data, sizeof(struct CPG)*n);
        if(q->_head > q->_tail) {
            for(i=q->_capacity-1; i>=q->_head; --i) {
                q->_data[i+delta].c = q->_data[i].c;
                q->_data[i+delta].g = q->_data[i].g;
                q->_data[i+delta].p = q->_data[i].p;
            }
        }
        q->_capacity = n;
    }
}

void Queue_push(QueuePtr q, int c, int p, int g)
{
    if(Queue_full(q))
        Queue_resize(q, q->_capacity*2);
    if(Queue_empty(q))
        q->_head = q->_tail = 0;
    else
        q->_tail = (q->_tail+1)%(q->_capacity);
    q->_data[q->_tail].c = c;
    q->_data[q->_tail].g = g;
    q->_data[q->_tail].p = p;
    q->_size++;
}

void Queue_pop(QueuePtr q)
{
    if(Queue_empty(q))
        return;
    if(q->_head == q->_tail)
        q->_head = q->_tail = -1;
    else
        q->_head = (q->_head+1)%(q->_capacity);
    q->_size--;
}

QueuePtr candidates;

#include <pthread.h>
int num_thread = 5;
pthread_mutex_t mut=PTHREAD_MUTEX_INITIALIZER;
pthread_t *thread;
const struct svm_problem *prob_thread=NULL;
const struct svm_parameter *para_thread=NULL;
int nfold_thread=0;
double (*eval_thread)(const double *act, const double *prd, int n)=NULL;

double best_c, best_g, best_p;
double best_eval=DBL_MIN;

double do_cv_and_update(double c, double g, double p, int verbose)
{
    int i, j, begin, end;
    int total = prob_thread->l;
    double value;
    int *fold_start = Malloc(int,nfold_thread+1);
    double *predictY = Malloc(double, total);
    struct svm_problem *train = Malloc(struct svm_problem, 1);
    struct svm_model *model;
    struct svm_parameter *para=copy_svm_parameter(para_thread);
    
    para->C = c;
    para->gamma = g;
    para->p = p;

    train->y = Malloc(double, total);
    train->x = Malloc(struct svm_node *, total);

    for(i=0; i<=nfold_thread; ++i)
        fold_start[i] = i*total/nfold_thread;
    for(i=0; i<nfold_thread; ++i) {
        begin = fold_start[i];
        end = fold_start[i+1];
        train->l = 0;
        for(j=0; j<begin; ++j) { // train
            train->y[train->l] = prob_thread->y[j];
            train->x[train->l] = prob_thread->x[j];
            train->l++;
        }
        for(j=end; j<total; ++j) { // train
            train->y[train->l] = prob_thread->y[j];
            train->x[train->l] = prob_thread->x[j];
            train->l++;
        }
        //printf("\ntraining set of %dth-fold:\n",i);
        //display_svm_problem(train);
        // train and predict
        model = svm_train(train,para);
        for(j=begin; j<end; ++j)
            predictY[j] = svm_predict(model,prob_thread->x[j]);
        svm_free_and_destroy_model(&model);
    }
    value = eval_thread(prob_thread->y, predictY, prob_thread->l);
    if(verbose) {
        printf("c=%g, g=%g, p=%g, r^2=%g\n", para->C, para->gamma, para->p, value);
        fflush(stdout);
    }
    
    pthread_mutex_lock(&mut);
    if(value > best_eval) {
        best_c = para->C;
        best_g = para->gamma;
        best_p = para->p;
        best_eval = value;
    }
    pthread_mutex_unlock(&mut);
    
    free(fold_start);
    free(predictY);
    free(train->y);
    free(train->x);
    free(train);
    svm_destroy_param(para);
    
    return value;
}

void *thread_cv(void *verbose)
{
    double c,p,g;
    double curr_eval;
    struct CPG *cpg;
    while(!Queue_empty(candidates)) {
        pthread_mutex_lock(&mut);
        cpg = Queue_head(candidates);
        if(cpg == NULL)
            break;
        c = pow(BASE_C, cpg->c);
        p = BASE_P * (cpg->p);
        g = pow(BASE_G, cpg->g);
        Queue_pop(candidates);
        pthread_mutex_unlock(&mut);
        if(*(int*)verbose) {
            printf("to check c=%g, g=%g, p=%g\n", c, g, p);
            fflush(stdout);
        }
        curr_eval = do_cv_and_update(c, g, p, *(int*)verbose);
    }
    pthread_exit(0);
}

double grid_search_cpg(const struct svm_problem *prob, struct svm_parameter *para,
        int nfold, int verbose, double (*eval)(const double *act, const double *pred, int n))
{
    int i, c, g, p;
    candidates = Queue_new(375);
    for(p=1; p<=5; ++p)
        for(g=-8; g<=8; ++g)
            for(c=-8; c<=2; ++c)
                Queue_push(candidates, c, p, g);
    prob_thread = prob;
    para_thread = para;
    nfold_thread = nfold;
    eval_thread = eval;
    best_c = para_thread->C; best_g = para_thread->gamma; best_p = para_thread->p;
    best_eval = DBL_MIN;
    //pthread_mutex_init(&mut, NULL);
    thread = (pthread_t *)malloc(sizeof(pthread_t)*num_thread);
    memset(thread, 0, sizeof(pthread_t)*num_thread);
    for(i=0; i<num_thread; ++i)
        pthread_create(&thread[i], NULL, thread_cv, &verbose);
    for(i=0; i<num_thread; ++i)
        if(thread[i] != 0)
            pthread_join(thread[i],NULL);

    para->C = best_c;
    para->gamma = best_g;
    para->p = best_p;
    if(verbose)
        printf("\n==> c=%g, g=%g, p=%g, r^2=%g\n", para->C, para->gamma, para->p, best_eval);
    Queue_free(candidates);
    //pthread_mutex_destroy(&mut);
    free(thread);
    thread = NULL;
    return best_eval;
}

#else
// `nfold`-fold CV will be used!
// eval: method to evaluate set of parameters. greater better
double grid_search_cpg(const struct svm_problem *prob, struct svm_parameter *para,
        int nfold, int verbose, double (*eval)(const double *act, const double *pred, int n))
{
    double best_eval = DBL_MIN;
    double curr_eval;
    double best_c=para->C, best_g=para->gamma, best_p=para->p;
    int i,j,k;
    //double *target = Malloc(double,prob->l);
    for(i=1; i<=5; ++i) {
        para->p = BASE_P * i;
        for(j=-8; j<=8; ++j) {
            para->gamma = pow(BASE_G,j);
            for(k=-8; k<=8; ++k) {
                para->C = pow(BASE_C,k);
                //svm_cross_validation(prob,para,nfold,target);
                //curr_eval = eval(prob->y,target,prob->l);
                curr_eval = svm_nCV(para,prob,nfold,eval);
                if(verbose)
                    printf("check c=%g,g=%g,p=%g => r^2=%g\n",para->C,para->gamma,para->p,curr_eval);
                if(curr_eval > best_eval) {
                    best_c = para->C;
                    best_g = para->gamma;
                    best_p = para->p;
                    best_eval = curr_eval;
                }
            }
        }
    }
    para->C = best_c;
    para->gamma = best_g;
    para->p = best_p;
    if(verbose)
        printf("==> c=%g, g=%g, p=%g, r^2=%g\n",para->C,para->gamma,para->p,best_eval);
    //free(target);
    return best_eval;
}
#endif

struct svm_problem *copy_and_refine_problem(const struct svm_problem *prob)
{
    int num_features;
    int i,j,k;
    struct svm_problem *newprob;

    // get feature names saved in `features`
    //num_features = extract_svm_features(prob,features);
    num_features = svm_num_features(prob);

    newprob = Malloc(struct svm_problem,1);
    newprob->l = prob->l;
    newprob->y = Malloc(double, newprob->l);
    newprob->x = Malloc(struct svm_node*, newprob->l);
    for(i=0; i<prob->l; ++i) {
        newprob->y[i] = prob->y[i];
        newprob->x[i] = Malloc(struct svm_node, num_features+1);
        k = 0; // index of prob->x[i]
        j = 0;
        for(; (j<num_features) && (prob->x[i][k].index != -1); ++j) {
            newprob->x[i][j].index = j+1;
            if(j+1 < prob->x[i][k].index)
                newprob->x[i][j].value = 0.;
            else {
                newprob->x[i][j].value = prob->x[i][k].value;
                ++k;
            }
        }
        for(; j<num_features; ++j) {
            newprob->x[i][j].index = j+1;
            newprob->x[i][j].value = 0.;
        }
        newprob->x[i][j].index = -1;
    }
    return newprob;
}


// usedd by `rfe`
struct FeaImport
{
    int index;
    double value;
};

int rfe_comp(const void *v1, const void *v2)
{
    const struct FeaImport *a1 = (const struct FeaImport *)v1;
    const struct FeaImport *a2 = (const struct FeaImport *)v2;
    if(a1->value < a2->value)
        return -1;
    else if(a1->value > a2->value)
        return 1;
    else {
        if(a1->index < a2->index)
            return -1;
        else
            return 1;
    }
}
// Parameter
// - calcImportance: function to evaluate each feature's relative importance.
//                 the second parameter should have already been alocated enough memory!!!
//                 if it's failure to estimate a feature's importance, then value of `DBL_MIN`
//                 should be given.
// - evaluate: function to evaluate actual and predicted Ys
void rfe(const struct svm_problem *prob, const struct svm_parameter *para,
        int step, int num_pick, int nfold, int search, double *importance,
        /*double *(*calcImportance)(const struct svm_problem *),*/
        double (*evaluate)(const double *act, const double *pred, int n))
{
    struct svm_problem *newprob;
    struct svm_parameter *newpara;
    struct FeaImport *fi;
    //double *importance;
    int *ranking, *support;
    int total_features,i,j,k,num_feature_remained,num_removed;
    //double *target = Malloc(double,prob->l);
    double value;

    // 1. preprocess feature importance
    total_features = svm_num_features(prob);
    num_feature_remained = total_features;
    //importance = calcImportance(prob);
    fi = Malloc(struct FeaImport, total_features);
    ranking = Malloc(int,total_features);
    support = Malloc(int,total_features);
    for(j=0; j<total_features; ++j) {
        fi[j].index = j;
        fi[j].value = importance[j];
        ranking[j] = 1;
        support[j] = 1;
    }
    // sort `fi` according to feature importance increasingly!!
    qsort(fi,total_features,sizeof(struct FeaImport),rfe_comp);
    //free(importance);

    // 2. evaluate the whole set of features (exclude those with feature importance being DBL_MIN)
    newprob = copy_part_svm_problem(prob,support);
    //display_svm_problem(newprob);
    newpara = copy_svm_parameter(para);
    printf("\nusing %d features\n", num_feature_remained);
    if(search == 1) {  // do grid search
        printf("do grid search... ");
        fflush(stdout);
        if(para->kernel_type==LINEAR && para->svm_type==EPSILON_SVR) {
            value = grid_search_cp(newprob,newpara,nfold,0,evaluate);
            printf(" {c=%g, p=%g} %g\n", newpara->C, newpara->p, value);
        }
        else if(para->kernel_type==RBF && para->svm_type==EPSILON_SVR) {
            value = grid_search_cpg(newprob,newpara,nfold,0,evaluate);
            printf(" {c=%g, g=%g, p=%g} %g\n",newpara->C, newpara->gamma, newpara->p, value);
        }
        else {
            fprintf(stderr,"Error: in `svmtools.c::rfe`, unsupport kernel_type or svm_type!\n");
            svm_destroy_param(newpara);
            exit(EXIT_FAILURE);
        }
    }
    else {  // do {nfold}-fold CV
        printf("do %d-fold CV.. ", nfold);
        //svm_cross_validation(newprob,newpara,nfold,target);
        //value = evaluate(newprob->y, target, newprob->l);
        value = svm_nCV(newpara,newprob,nfold,evaluate);
        printf("=> CV result: %g\n",value);
    }
    fflush(stdout);
    svm_destroy_param(newpara);
    free_svm_problem1(newprob);

    // 3. begin to remove `k` features each time.
    while(num_feature_remained > num_pick) {
        if(step > (num_feature_remained - num_pick))
            k = num_feature_remained - num_pick;
        else
            k = step;
        // to remove `k` features by updating `support`
        num_removed = total_features - num_feature_remained;
        for(i=num_removed; i<(num_removed+k); ++i)
            support[fi[i].index] = 0;
        num_feature_remained -= k;
        // update `ranking`
        for(i=0; i<total_features; ++i)
            if(support[i] == 0)
                ranking[i] += 1;
        newprob = copy_part_svm_problem(prob,support);
        newpara = copy_svm_parameter(para);
        printf("\nusing %d features\n", num_feature_remained);
        if(search == 1) {  // do grid search
            printf("do grid search... ");
            fflush(stdout);
            if(para->kernel_type==LINEAR && para->svm_type==EPSILON_SVR) {
                value = grid_search_cp(newprob,newpara,nfold,0,evaluate);
                printf(" {c=%g, p=%g} %g\n", newpara->C, newpara->p, value);
            }
            else if(para->kernel_type==RBF && para->svm_type==EPSILON_SVR) {
                value = grid_search_cpg(newprob,newpara,nfold,0,evaluate);
                printf(" {c=%g, g=%g, p=%g} %g\n",newpara->C, newpara->gamma, newpara->p, value);
            }
            else {
                fprintf(stderr,"Error: in `svmtools.c::rfe`, unsupport kernel_type or svm_type!\n");
                svm_destroy_param(newpara);
                exit(EXIT_FAILURE);
            }
        }
        else {  // do {nfold}-fold CV
            printf("do %d-fold CV... ", nfold);
            //svm_cross_validation(newprob,newpara,nfold,target);
            //value = evaluate(newprob->y, target, newprob->l);
            value = svm_nCV(newpara,newprob,nfold,evaluate);
            printf("=> CV result: %g\n",value);
        }
        fflush(stdout);
        svm_destroy_param(newpara);
        free_svm_problem1(newprob);
    }
    // END of while-loop
    
    printf("\nranking: ");
    for(i=0; i<total_features; ++i)
        printf(" %d",ranking[i]);
    printf("\n");

    free(fi);
    free(support);
    free(ranking);
    //free(target);

}
