#include "network.h"

using namespace std;

Network network;

int main()
{

	mkl_set_num_threads(4);

	network.SGD(0.97,50);

	char s[8000];
	double test[innode];
	int t;
	FILE* fp;
	FILE* fpout;
	fopen_s(&fpout,"submit.csv", "w");
	fopen_s(&fp, "test.csv", "r");
	if (fp == NULL)
	{
		cout << "No test.csv" << endl;
		return 0;
	}

	fgets(s, sizeof(s), fp);
	fprintf(fpout, "ImageId,Label\n");
	for (int i = 1; i <= TEST_NUM; i++)
	{
		int classified;
		memset(test, 0, sizeof(double)*innode);
		for (int j = 0; j < innode; j++)
		{
			fscanf_s(fp, "%d", &t);
			fgetc(fp);	//, or \n
			if (t > 0)
				test[j] = 1;
		}

		classified = network.recognize(test);
		//print_image(test);
		//cout << endl << classified << endl;

		fprintf(fpout, "%d,%d\n", i, classified);

		//getchar();
		if (i % 100 == 0)
			cout << i << endl;
	}



	return 0;
}