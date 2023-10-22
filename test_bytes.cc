#include "learned_index.h"
#include "plr.h"
#include <iostream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <fstream>
using namespace std;
using namespace chrono;
// #define INTERNAL

int main(int argc,char *argv[]){

  RMIConfig rmi_config;
  RMIConfig::StageConfig first, second;

  first.model_type = RMIConfig::StageConfig::LinearRegression;
  first.model_n = 1;

  
  second.model_n = 1000;
  second.model_type = RMIConfig::StageConfig::LinearRegression;
  rmi_config.stage_configs.push_back(first);
  rmi_config.stage_configs.push_back(second);

  LearnedRangeIndexSingleKey<uint64_t,float> table(rmi_config);

  srand((unsigned)time(NULL));
  const uint32_t M = 1000000;
  const uint32_t num_op = 0.7 * M;
  vector<double> x;
  vector<double> y;
  double key = 0;
  double value;
  srand((unsigned) time(nullptr));
  ifstream ifs("/home/yjn/Desktop/VLDB/Dataset/fb_seq_data.csv");

  for (uint32_t i = 0; i < num_op; i += 1) {
    ifs >> key;
    value = 64;
    table.insert(key, value);
    x.push_back(key);
    y.push_back(value);
  }
  ifs.close();
  std::cout << "Finish Insert" << std::endl;

  const double PLR_ERROR = 8;
  PLR plr(PLR_ERROR);
  auto st_time = high_resolution_clock::now();
  std::vector<Segment> segs = std::move(plr.train(x));
  auto en_time = high_resolution_clock::now();
  auto duration = duration_cast<nanoseconds>(en_time - st_time).count();
  std::cout << "PLR training time: " << duration << " ns" << std::endl;
  Segment last_seg(x.back(), 0, x.size() - 1, x.back());
  segs.emplace_back(last_seg);
  uint64_t duration_1 = 0, duration_2 = 0, duration_3 = 0, duration_4 = 0;
  st_time = high_resolution_clock::now();
  for (int i = 0; i < x.size(); i++){
    key = x[i];
    uint32_t left = 0, right = (uint32_t) segs.size() - 1;
#ifdef INTERNAL
    auto st_time_1 = high_resolution_clock::now();
#endif
    while (left != right - 1) {
      uint32_t mid = (right + left) / 2;
      if (key < segs[mid].x_) right = mid;
      else left = mid;
    }
    if (key > segs[left].x2_) {
      assert(left != string_segments.size() - 2);
      ++left;
      key = segs[left].x_;
    }
#ifdef INTERNAL
    auto en_time_1 = high_resolution_clock::now();
    duration_1 += duration_cast<nanoseconds>(en_time_1 - st_time_1).count();
    auto st_time_2 = high_resolution_clock::now();
#endif
    double result = key * segs[left].k_ + segs[left].b_;
#ifdef INTERNAL
    auto en_time_2 = high_resolution_clock::now();
    duration_2 += duration_cast<nanoseconds>(en_time_2 - st_time_2).count();
#endif
    uint64_t lower = result - PLR_ERROR > 0 ? (uint64_t) std::floor(result - PLR_ERROR) : 0;
    uint64_t upper = (uint64_t) std::ceil(result + PLR_ERROR);
#ifdef INTERNAL
    auto st_time_3 = high_resolution_clock::now();
#endif
    while (lower < upper) {
      uint64_t mid = (lower + upper) / 2;
      if (key == x[mid]) break;
      else if (key < x[mid]) upper = mid - 1;
      else lower = mid + 1;
    }
#ifdef INTERNAL
    auto en_time_3 = high_resolution_clock::now();
    duration_3 += duration_cast<nanoseconds>(en_time_3 - st_time_3).count();
#endif
  }
  en_time = high_resolution_clock::now();
  duration_4 = duration_cast<nanoseconds>(en_time - st_time).count();
  std::cout << "binary search model: " << duration_1 / x.size() << std::endl;
  std::cout << "linear calculation: " << duration_2 / x.size() << std::endl;
  std::cout << "binary search last mile: " << duration_3 / x.size() << std::endl;
  std::cout << "avg search latency: " << duration_4 / x.size() << std::endl;
//  std::cout << "binary search model: " << duration_1 << std::endl;
//  std::cout << "linear calculation: " << duration_2 << std::endl;
//  std::cout << "binary search last mile: " << duration_3 << std::endl;
return 0;



  st_time = high_resolution_clock::now();
  table.finish_insert();
  table.finish_train();
  en_time = high_resolution_clock::now();
  duration = duration_cast<nanoseconds>(en_time - st_time).count();
  std::cout << "OLS Training time: " << duration << " ns" << std::endl;

  for (int i = 0; i < x.size(); i++){
    key = x[i];
    value = y[i];
    auto value_get = table.get(key);
    double bit = 1.0 * (value-value_get) / value;
    int block = value_get / 4096;
    // cout << i << " result: " << value_get << " : " << value << "; block:" << block <<
    //       "; error:" << (value-value_get)  << ";error bit: "<< bit << endl;
  }
  // for (int i = 0; i < result.size(); i++){
  //   if (result[i] != 0)
  //     cout << i << " block_num: " << result[i] << endl;
  // }

  table.printR();

  // serialize && deserialize
  string param;
  table.serialize(param);
  // cout << "serialize: " << param << " ;lenth: " << param.length() << endl;
  cout << "serialize lenth: " << param.length() << endl;

  LearnedRangeIndexSingleKey<uint64_t,float> Rtable(param, rmi_config);
  int find=0, no_find=0;
  for (int i = 0; i < x.size(); i++){
    auto value_get = table.get(x[i]);
    auto Rvalue_get = Rtable.get(x[i]);
    
    if (value_get != Rvalue_get){
//      cout << i << ": value_get( " << value_get <<  " )!= Rvalue_get( " << Rvalue_get << " )" << endl;
      no_find ++;
    }
    else{
//      cout << i << ": value_get( " << value_get <<  " )== Rvalue_get( " << Rvalue_get << " )" << endl;
      find ++;
    }
  }
  cout << "total: " << x.size() << " ;right: " << find << " ;wrong: " << no_find << endl;
} // end namespace test
