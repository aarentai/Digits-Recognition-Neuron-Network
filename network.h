#include<mkl.h>
#include<iostream>
#include<cstdio>
#include<math.h>
#include<algorithm>
#include<time.h>
#include<random>

const int innode = 784;			//number of input node
const int hiddennode = 300;		//number of hidden node
const int hiddenlayer = 1;		//number of hiddenlayer
const int outnode = 10;			//number of output node
#define IMG_SIZE 28
#define TRAIN_NUM 42000//Total samples
#define VER_NUM 2000
#define TEST_NUM 28000

using namespace std;

default_random_engine rd(time(NULL));
mt19937 gen(rd());
normal_distribution<> nor_dis(0.0, 0.1);


struct inputData
{
	double train[innode];
	double label[outnode];
};

struct Layer
{
	double* value;
	double* weight;//matrix
	double* bias;

	double* deltaw;//matrix sum of delta weight
	double* deltab;//sum of delta bias
	double* delta;//sensitivity
};

void print_image(double *img)
{
	for (int j = 0; j < innode; j++)
	{
		if (j%IMG_SIZE == 0)
			printf("\n");
		if (img[j] == 1.0)
			printf("1");
		else
			printf("0");
	}
}
void set_ran_num(double* a, int n)
{
	for (int i = 0; i < n; i++)
		a[i] = nor_dis(gen);
	//a[i] = ((2.0*(double)rand() / RAND_MAX) - 1);
}

double dot_product(double* a, double* b,int n)
{
	double sum = 0.0;
	for (int i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

void matrix_vector(double* m, double* v,double* result, int row, int col)//Ax
{
	for (int i = 0; i < row; i++)
	{
		result[i] = dot_product(m+i*col, v, col);
	}
}

void matrix_vector(double* m, double* v, double *b, double* result, int row, int col)//Ax+b
{
	for (int i = 0; i < row; i++)
	{
		result[i] = dot_product(m + i*col, v, col)+b[i];
	}
}

inline double sigmoid(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

void sigmoid_vec(double* a,int n)
{
	for (int i = 0; i < n; i++)
		a[i] = sigmoid(a[i]);
}

inline double d_sigmoid(double sigmoid_value)//Derivative of the sigmoid function
{
	return sigmoid_value*(1 - sigmoid_value);
}


class Network
{
public:
	Network();
	~Network();
	void forward(double* input);//length is innode
	void getinputdata();
	void backPropagation(int index);
	void SGD(double threshold, int epochmax);//stochastic gradient descent(train)
	void train_mini_batch(int* index);
	double get_error(int index);
	int recognize(double* test);
	void updata_w_b();
	void calc_per();
private:
	double error = 0;
	double percentage;
	double learningRate =1.5;		//learningRate

	Layer layer[hiddenlayer + 1];

	inputData inputdata[TRAIN_NUM] = {};
	int mini_batch_size = 10;
	int mini_batch_index[TRAIN_NUM];

	int epoch = 0;

};


void Network::getinputdata()
{
	int t;
	char s[8000];
	FILE* fp;
	fopen_s(&fp,"train.csv", "r");
	if (fp == NULL)
	{
		cout << "No train.csv" << endl;
		exit(0);
	}

	fgets(s, sizeof(s), fp);

	for (int i = 0; i < TRAIN_NUM; i++)
	{
		fscanf_s(fp, "%d", &t);
		fgetc(fp);	//, or \n
		inputdata[i].label[t] = 1.0;
		for (int j = 0; j < innode; j++)
		{
			fscanf_s(fp, "%d", &t);
			fgetc(fp);	//, or \n
			if (t > 0)//Change 0 to 1
				inputdata[i].train[j] = 1.0;
		}
		//print_image(inputdata[i].train);
		//cout << endl;
		//for (int j = 0; j < 10; j++)
		//	cout << inputdata[i].label[j];
		//getchar();
		if (i % 100 == 0)
			cout << i << endl;
	}
	fclose(fp);
}
Network::Network()
{
	//inputlayer
	getinputdata();

	for (int i = 0; i < TRAIN_NUM- VER_NUM; i++)
		mini_batch_index[i] = i;
	srand(time(NULL));


	for (int i = 0; i < hiddenlayer; i++)
	{
		layer[i].bias = new double[hiddennode];
		layer[i].value = new double[hiddennode];
		layer[i].delta = new double[hiddennode];
		layer[i].deltab = new double[hiddennode];
		if (i == 0)//Set the number to innode(input layer)
		{
			layer[i].weight = new double[hiddennode*innode];
			layer[i].deltaw = new double[hiddennode*innode];
			set_ran_num(layer[i].weight, hiddennode*innode);
		}
		else//Set the number to hiddennode(previous hidden layer)
		{
			layer[i].weight = new double[hiddennode*hiddennode];
			layer[i].deltaw = new double[hiddennode*hiddennode];
			set_ran_num(layer[i].weight, hiddennode*hiddennode);
		}
		memset(layer[i].bias, 0,sizeof(double)*hiddennode);
		//set_ran_num(layer[i].bias, hiddennode);
	}
	//The output layer
	layer[hiddenlayer].bias = new double[outnode];
	layer[hiddenlayer].value = new double[outnode];
	layer[hiddenlayer].deltab = new double[outnode];
	layer[hiddenlayer].delta = new double[outnode];
	layer[hiddenlayer].weight = new double[outnode*hiddennode];
	layer[hiddenlayer].deltaw = new double[outnode*hiddennode];
	memset(layer[hiddenlayer].bias, 0, sizeof(double)*outnode);
	//set_ran_num(layer[hiddenlayer].bias, outnode);
	set_ran_num(layer[hiddenlayer].weight, outnode*hiddennode);

}

Network::~Network()
{
//Too lazy to release memory
}

void Network::forward(double* input)
{
	//length is innode
	double* in;
	int row, col;
	for (int i = 0; i <= hiddenlayer; i++)//hidden layer and the output layer
	{
		if (i == 0)//first hidden layer
		{
			in = input;
			row = hiddennode;
			col = innode;
		}
		else if(i== hiddenlayer)//the output layer
		{
			in = layer[i - 1].value;//the previous  hidden layer
			row = outnode;
			col = hiddennode;
		}
		else
		{
			in = layer[i - 1].value;//the previous  hidden layer
			row = hiddennode;
			col = hiddennode;
		}
		memcpy(layer[i].value, layer[i].bias, sizeof(double)*row);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, row, col, 1, layer[i].weight, col, in, 1, 1, layer[i].value, 1);
		//matrix_vector(layer[i].weight, in, layer[i].bias, layer[i].value, row, col);
		sigmoid_vec(layer[i].value, row);
	}

}

double Network::get_error(int index)
{
	double err[outnode];
	vdSub(outnode, layer[hiddenlayer].value, inputdata[index].label, err);
	return cblas_ddot(outnode, err, 1, err, 1)/2.0;
}
void Network::train_mini_batch(int* index)
{
	for (int i = hiddenlayer; i >= 0; i--)
	{
		if (i == hiddenlayer)//outputlayer
		{
			memset(layer[i].delta, 0, sizeof(double)*outnode);
			memset(layer[i].deltab, 0, sizeof(double)*outnode);
			memset(layer[i].deltaw, 0, sizeof(double)*hiddennode*outnode);
		}
		else
		{
			memset(layer[i].delta, 0, sizeof(double)*hiddennode);
			memset(layer[i].deltab, 0, sizeof(double)*hiddennode);
			memset(layer[i].deltaw, 0, sizeof(double)*hiddennode*innode);
		}
	}

	for (int i = 0; i < mini_batch_size; i++)
	{
		forward(inputdata[index[i]].train);
		error += get_error(index[i]);
		backPropagation(index[i]);
	}
	updata_w_b();
}


void Network::SGD(double threshold,int epochmax)//train
{
	int i = 0;

	while (percentage<threshold&&epoch<epochmax)
	{
		train_mini_batch(mini_batch_index+i);
		
		i += mini_batch_size;
		if (i > TRAIN_NUM- VER_NUM)
		{
			i %= (TRAIN_NUM- VER_NUM);
			random_shuffle(mini_batch_index, mini_batch_index + TRAIN_NUM-VER_NUM);
			epoch++;
			calc_per();
			cout << "Error sum:" << error << endl;
			error = 0;
			if (epoch>=30&& epoch%10==0)
				learningRate *= 0.8;
		}
		
	}
}

int Network::recognize(double* test)
{
	forward(test);
	return (int)cblas_idamax(outnode, layer[hiddenlayer].value, 1);
}

void Network::backPropagation(int index)
{
	for (int i = hiddenlayer; i >= 0; i--)
	{
		if (i == hiddenlayer)//outputlayer
		{
			for (int j = 0; j < outnode; j++)
			{
				layer[i].delta[j] = (layer[i].value[j] - inputdata[index].label[j])*d_sigmoid(layer[i].value[j]);
				layer[i].deltab[j] += layer[i].delta[j];
				for (int k = 0; k < hiddennode; k++)
					layer[i].deltaw[j*hiddennode + k] += layer[i].delta[j] * layer[i - 1].value[k];
			}
		}
		else if (i == 0)//the first hidden layer
		{
			memset(layer[i].delta, 0, sizeof(double)*hiddennode);
			for (int j = 0; j < hiddennode; j++)
			{
				for (int k = 0; k < outnode; k++)//change if hiddenlayer>1
					layer[i].delta[j] += layer[i + 1].delta[k] * layer[i + 1].weight[k*hiddennode + j];
				layer[i].delta[j] *= d_sigmoid(layer[i].value[j]);
				layer[i].deltab[j] += layer[i].delta[j];
				for (int k = 0; k < innode; k++)
					layer[i].deltaw[j*innode + k] += layer[i].delta[j] * inputdata[index].train[k];
			}
		}
		else
		{
			//Not useful when hiddenlayer==1
			cout << "Error" << endl;
		}
	}
}

void Network::updata_w_b()
{
	for (int i = hiddenlayer; i >= 0; i--)
	{
		if (i == hiddenlayer)//outputlayer
		{
			for (int j = 0; j < outnode; j++)
			{
				layer[i].bias[j] -= learningRate*layer[i].deltab[j]/mini_batch_size;
				for (int k = 0; k < hiddennode; k++)
					layer[i].weight[j*hiddennode + k] -= learningRate*layer[i].deltaw[j*hiddennode + k] / mini_batch_size;
			}
		}
		else if (i == 0)//the first hidden layer
		{
			for (int j = 0; j < hiddennode; j++)
			{
				layer[i].bias[j] -= learningRate*layer[i].deltab[j] / mini_batch_size;
				for (int k = 0; k < innode; k++)
					layer[i].weight[j*innode + k] -= learningRate*layer[i].deltaw[j*innode + k] / mini_batch_size;
			}
		}
		else
		{
			//Not useful when hiddenlayer==1
			cout << "Noooooooooo" << endl;
		}
	}
}


void Network::calc_per()
{
	int cnt = 0;

	for (int i = 0; i < TRAIN_NUM - VER_NUM; i++)
	{
		forward(inputdata[i].train);
		int a = cblas_idamax(outnode, layer[hiddenlayer].value, 1);
		int b = cblas_idamax(outnode, inputdata[i].label, 1);
		if (a == b)
			cnt++;
	}
	cout << "Epoch: " << epoch << " Training Correct count: " << cnt << '/' << (TRAIN_NUM - VER_NUM) << endl;

	cnt = 0;
	for (int i = TRAIN_NUM-VER_NUM; i < TRAIN_NUM; i++)
	{
		forward(inputdata[i].train);
		int a = cblas_idamax(outnode, layer[hiddenlayer].value, 1);
		int b = cblas_idamax(outnode, inputdata[i].label, 1);
		//print_image(inputdata[i].train);
		//cout << a << ' ' << b << endl;
		//for (int j = 0; j <= 9; j++)
		//	cout << layer[hiddenlayer].value[j] << ' ';
		if (a==b)
			cnt++;
	}
	cout << "Epoch: "<<epoch<<" Testing Correct count: " << cnt << '/' << VER_NUM << endl;
	percentage = (double)cnt / VER_NUM;

}