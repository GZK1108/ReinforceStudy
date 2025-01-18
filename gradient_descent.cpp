#include "src.h"

using namespace std;




vector<vector<double>> generate_data(int num_points, int radius) {
	vector<vector<double>> data(num_points, vector<double>(2, 0));
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dist(-radius/2, radius/2);
	for (int i = 0; i < num_points; ++i) {
		data[i][0] = dist(gen);
		data[i][1] = dist(gen);
	}
	return data;
}


//随机梯度下降
pair<vector<vector<double>>, vector<double>> sgd(vector<vector<double>> data, vector<double> w, vector<double> wstar, double learning_rate=0.001) {
	vector<vector<double>> wlist;
	wlist.push_back(w);
	vector<double> errors;
	double error = sqrt((w[0] - wstar[0]) * (w[0] - wstar[0]) + (w[1] - wstar[1]) * (w[1] - wstar[1]));
	errors.push_back(error);
	for (int k = 0; k < data.size(); ++k) {
		learning_rate = 1.0 / (k+1);

		w[0] = w[0] - learning_rate * (w[0] - data[k][0]);
		w[1] = w[1] - learning_rate * (w[1] - data[k][1]);
		wlist.push_back(w);
		error = sqrt((w[0] - wstar[0]) * (w[0] - wstar[0]) + (w[1] - wstar[1]) * (w[1] - wstar[1]));
		errors.push_back(error);
	}
	return { wlist,errors };
}


//小批量梯度下降
pair<vector<vector<double>>, vector<double>> mbgd(vector<vector<double>> data, vector<double> w, vector<double> wstar, int batch = 10, double learning_rate = 0.001) {
	vector<vector<double>> wlist;
	wlist.push_back(w);
	vector<double> errors;
	double error = sqrt((w[0] - wstar[0]) * (w[0] - wstar[0]) + (w[1] - wstar[1]) * (w[1] - wstar[1]));
	errors.push_back(error);
	for (int k = 0; k < data.size() / batch; ++k) {
		learning_rate = 1.0 / (k+1);
		int start = k * batch;
		int end = (k + 1) * batch;
		int sum0 = 0, sum1 = 0;
		// 更新权重
		for (int i = start; i < end; ++i) {
			sum0 += w[0] - data[i][0];
			sum1 += w[1] - data[i][1];
		}
		w[0] = w[0] - learning_rate * sum0 / batch;
		w[1] = w[1] - learning_rate * sum1 / batch;
		wlist.push_back(w);
		error = sqrt((w[0] - wstar[0]) * (w[0] - wstar[0]) + (w[1] - wstar[1]) * (w[1] - wstar[1]));
		errors.push_back(error);
	}
	return { wlist,errors };
}

//绘制散点图
void drawScatter(vector<vector<double>> data, std::initializer_list<vector<vector<double>>> lst, string labels[]) {
	vector<double> x, y;
	for (int i = 0; i < data.size(); ++i) {
		x.push_back(data[i][0]);
		y.push_back(data[i][1]);
	}
	plt::scatter(x, y, 5.0, { {"color", "w"},{"marker", "o"},{"edgecolors","b"} });
	
	//绘制直线
	for (auto wlist = lst.begin(); wlist != lst.end(); ++wlist) {
		vector<double> xline, yline;
		for (int i = 0; i < wlist->size(); ++i) {
			xline.push_back((*wlist)[i][0]);
			yline.push_back((*wlist)[i][1]);
		}
		plt::plot(xline, yline, { {"label", labels[wlist - lst.begin()]},{"linewidth","1.0"},{"marker","*"},{"markersize","2.0"} });

	}
	//原点
	vector<double> x0(1, 0), y0(1, 0);
	plt::scatter(x0, y0, 85.0, { {"color", "w"},{"marker", "o"},{"edgecolors","k"} });
	plt::legend();
	plt::show();
}


//绘制误差
void drawError(std::initializer_list<vector<double>> lst, string labels[]) {
	int len = (lst.end() - 1)->size();
	plt::xlim(0, 10);
	vector<double> x;
	for (int i = 0; i < len; ++i) {
		x.push_back(i);
	}
	for (auto errors = lst.begin(); errors != lst.end(); ++errors) {
		vector<double> y(errors->begin(), errors->end() - 1);
		plt::plot(y, { {"label", labels[errors - lst.begin()]}, {"marker","s"},{"markersize","1.0"}});
	}
	plt::legend();
	plt::show();
}

/*
int main() {
	vector<vector<double>> data = generate_data(400, 30);
	vector<double> w = { 50, 50 };
	vector<double> wstar = { 0, 0 };
	//pair<vector<vector<double>>, vector<double>> result_sgd = sgd(data, w, wstar, 0.005);
	pair<vector<vector<double>>, vector<double>> result_mbgd = mbgd(data, w, wstar, 1);
	pair<vector<vector<double>>, vector<double>> result_mbgd_10 = mbgd(data, w, wstar, 10);
	pair<vector<vector<double>>, vector<double>> result_mbgd_50 = mbgd(data, w, wstar, 50);
	pair<vector<vector<double>>, vector<double>> result_mbgd_100 = mbgd(data, w, wstar, 100);
	string labels[] = { "SGD m=1","MBGD m=10","MBGD m=50" ,"MBGD m=100" };
	drawScatter(data, { result_mbgd.first,result_mbgd_10.first,result_mbgd_50.first,result_mbgd_100.first }, labels);
	drawError({ result_mbgd.second,result_mbgd_10.second,result_mbgd_50.second,result_mbgd_100.second }, labels);
	return 0;
}
*/