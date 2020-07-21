#include <iostream>
#include <assert.h>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <unordered_map>
#include <iterator>
#include <sstream>

#define LINE_BUF_SIZE 1024*1024
#define VEC_SIZE 16

using namespace std;

struct UserFollow {
	long uid;
	vector<long> vec_follow;
};

typedef unordered_map<long, vector<float>> USER_VEC_MAP;
typedef vector<UserFollow> USER_FOLLOW_VEC;

template <typename T>
string to_stringr(const T& val) {
	stringstream ss;
	ss<<val;
	string ret;
	ss>>ret;
	ss.clear();
	return ret;
}

void  splitEx(const string& str, const string& delim, vector<string>& vec_out) {
    if("" == str) return;

    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p) {
        string s = p;
        vec_out.push_back(s); 
        p = strtok(NULL, d);
    }
}


bool load_user_vec(const string& file_name, USER_VEC_MAP& user_vec_map) {
	char str[LINE_BUF_SIZE];
	ifstream fin(file_name);
	if (!fin.is_open()) {
		cout << "open file error " << file_name << endl;
		return false;
	}

	vector<string> vec_line;
	vector<string> vec_value;
	vector<float> vec_tmp;
	while(fin.getline(str, LINE_BUF_SIZE)) {
		vec_line.clear();
		splitEx(str, "\t", vec_line);
		if (vec_line.size() == 2) {
			vec_value.clear();
			vec_tmp.clear();
			splitEx(vec_line[1], ",", vec_value);
			for (int i = 0; i < vec_value.size(); i++) {
				vec_tmp.push_back(atof(vec_value[i].c_str()));
			}
			if (user_vec_map.find(atol(vec_line[0].c_str())) == user_vec_map.end()) {
				user_vec_map[atol(vec_line[0].c_str())] = vec_tmp;
			}
		} else {
			cout << "head line error: " << str << endl;
		}
	}

	fin.close();
	return true;
}

bool load_user_follow(const string& file_name, USER_FOLLOW_VEC& vec_user_follow) {
	char str[LINE_BUF_SIZE];
	ifstream fin(file_name);
	if (!fin.is_open()) {
        cout << "open file error " << file_name << endl;
        return false;
    }
	
	vector<string> vec_line;
    vector<string> vec_value;
	UserFollow tmp_follow;
	while(fin.getline(str, LINE_BUF_SIZE)) {
		vec_line.clear();
		splitEx(str, "\t", vec_line);
		if (vec_line.size() == 2) {
			tmp_follow.uid = atol(vec_line[0].c_str());
			tmp_follow.vec_follow.clear();
			vec_value.clear();
			splitEx(vec_line[1], ",", vec_value);
			for (int i = 0; i < vec_value.size() && i < 200; i++) {
				tmp_follow.vec_follow.push_back(atol(vec_value[i].c_str()));
			}
			if (tmp_follow.vec_follow.size() > 0) {
				vec_user_follow.push_back(tmp_follow);
			} else {
				cout << "follow size 0: " << str << endl;	
			}
		} else {
			 cout << "follow line error: " << str << endl;
		}
	}

	return true;
}

void get_mean_vector(USER_VEC_MAP& map_head_vec, USER_FOLLOW_VEC& vec_user_follow, map<long, string>& map_user_vec) {
	for (int i = 0; i < vec_user_follow.size(); i++) {
		vector<float> vec_sum;
		vec_sum.clear();
		vec_sum.resize(VEC_SIZE);
		for (int j = 0; j < vec_user_follow[i].vec_follow.size(); j++) {
			long &followid = vec_user_follow[i].vec_follow[j];
			USER_VEC_MAP::iterator it = map_head_vec.find(followid);
			if (it != map_head_vec.end()) {
				for (int i = 0; i < VEC_SIZE; i++) {
					vec_sum[i] += it->second[i];
				}
			}
		}
		
		string str_vec_list;
		str_vec_list.clear();
		for (int k = 0; k < VEC_SIZE; k++) {
			vec_sum[k] /= vec_user_follow[i].vec_follow.size();
			if (k == 0) {
				str_vec_list += to_string(vec_sum[k]);
			} else {
				str_vec_list += "," + to_string(vec_sum[k]);
			}
		}
		map_user_vec[vec_user_follow[i].uid] = str_vec_list;
	}	
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		cout << "input args error: argc must be 4!" << endl;
		return -1;
	}

	// 获取参数
	string file_head_vec = argv[1];
	string file_follow = argv[2];
	string file_output = argv[3];
	cout << "file_follow="  << file_follow << ", file_head_vec=" << file_head_vec << ", file_output=" << file_output  << endl;

	// 加载大v账号模型向量
	USER_VEC_MAP map_head_vec;
	if (!load_user_vec(file_head_vec, map_head_vec)) {
		cout << "load user vector error!" << endl;
		return -1;
	}
	cout << "head user size: " << map_head_vec.size() << endl;

	// 读取followlist到内存
	USER_FOLLOW_VEC vec_user_follow;
	if (!load_user_follow(file_follow, vec_user_follow)) {
		cout << "load user follow error!" << endl;
		return -1;
	}
	cout << "user follow size: " << vec_user_follow.size() << endl;

	// 计算平均向量
	map<long, string> map_user_vec;
	get_mean_vector(map_head_vec, vec_user_follow, map_user_vec);

	// 打印用户向量
	ofstream of(file_output);
	map<long, string>::iterator it = map_user_vec.begin();
	for (; it != map_user_vec.end(); it++) {
		of << it->first << "\t" << it->second << endl;
	}	
	of.clear();
	of.close();
	
	return 0;
}
