#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <climits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "marshal.hpp"

#if !defined(COUT_THIS)
#define COUT_THIS(this) std::cout << this << std::endl
#endif  // COUT_THIS

#if !defined(MODEL_H)
#define MODEL_H

#define MKL_MALLOC_ALIGN 64


typedef int64_t learned_addr_t;

template <class D>
inline void min_max(const std::vector<D> &vals, D &max, D &min) {
  assert(vals.size() != 0);

  max = vals[0];
  min = vals[0];
  for (const D &val : vals) {
    if (val > max) max = val;
    if (val < min) min = val;
  }
}

template <class D>
inline void mean(const std::vector<D> &vals, D &mean) {
  assert(vals.size() != 0);

  double sum = 0;
  for (const D &val : vals) {
    sum += val;
  }

  mean = sum / vals.size();
}

template <class Model_T>
bool prepare_last_helper(Model_T *model, const std::vector<double> &keys,
                         const std::vector<learned_addr_t> &indexes) {
  double not_used, not_used_either;
  model->prepare(keys, indexes, not_used, not_used_either);

  std::vector<int64_t> errors;
  for (int i = 0; i < keys.size(); ++i) {
    double key = keys[i];
    int64_t index_actual = indexes[i];
    //int64_t index_pred = static_cast<int>(model->predict(key));
    /**
     * According to https://stackoverflow.com/questions/9695329/c-how-to-round-a-double-to-an-int,
     * using std::round is a more stable way to round.
     */
    int64_t index_pred = std::round(model->predict(key));
    errors.push_back(index_actual - index_pred);
    //if(key == 23002729) {
    //      printf("check inserted predict: %d, actual : %d\n",index_pred,index_actual);
    //}
  }
  return true;
}

template <class Model_T>
inline void predict_last_helper(Model_T *model, const double key, learned_addr_t &pos) {
  pos = std::round(model->predict(key));
}

class BestMapModel {
  public:
    void prepare(const std::vector<double> &keys,
                const std::vector<learned_addr_t> &indexes, double &index_pred_max, double &index_pred_min) {
      if (keys.size() == 0) return;

      key_size = keys.size();
      assert(keys.size() == indexes.size());

      for (uint32_t i = 0; i < key_size; ++i) {
        key_index[keys[i]] = indexes[i];
      }

      std::vector<double> index_preds;
      for (const double key: keys) {
        index_preds.push_back(predict(key));
      }
      min_max(index_preds, index_pred_max, index_pred_min);
    }

    double predict(const double key) { return (double)key_index[key]; }

    inline void prepare_last(const std::vector<double> &keys,
                            const std::vector<learned_addr_t> &indexes) {
      prepare_last_helper<BestMapModel>(this, keys, indexes);
    }


  private:
    std::map<double, learned_addr_t> key_index;
    uint64_t key_size;
};

#define REPORT_TNUM 1
class LinearRegression {
 public:
  void prepare(const std::vector<double> &keys,
               const std::vector<learned_addr_t> &indexes, double &index_pred_max, double &index_pred_min) {
    std::set<double> unique_keys;
    //printf("model prepare num: %lu\n",keys.size());
    for (double key: keys) {
      unique_keys.insert(key);
    }

    if (unique_keys.size() == 0) return;

    if (unique_keys.size() == 1) {
      bias = indexes[0];
      w = 0;
      return;
    }

    uint32_t count_ = keys.size();
    double x_sum_ = 0, y_sum_ = 0, xx_sum_ = 0, xy_sum_ = 0;
    for (uint32_t i = 0; i < count_; i += 1) {
      x_sum_ += keys[i];
      y_sum_ += static_cast<double>(indexes[i]);
      xx_sum_ += keys[i] * keys[i];
      xy_sum_ += keys[i] * indexes[i];
    }
    auto slope = static_cast<long double>(
            (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
            (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<long double>(
            (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
    bias = intercept;
    w = slope;

    std::vector<double> index_preds;
    for (const double key: keys) {
      index_preds.push_back(predict(key));
    }
    min_max(index_preds, index_pred_max, index_pred_min);
  }

  double predict(const double key) {
    auto res = bias + w * key;
    // std::cout << "predixt:  " << res << "; using key: " << key << "; bias: " << bias << "; w: " << w << std::endl;
    return std::max(res,0.0); // avoid 0 overflow
  }

  inline bool prepare_last(const std::vector<double> &keys,
                           const std::vector<learned_addr_t> &indexes) {
    return prepare_last_helper<LinearRegression>(this, keys, indexes);
  }

  inline void predict_last(const double key, learned_addr_t &pos) {
    predict_last_helper<LinearRegression>(this, key, pos);
  }

 public:

  static mousika::Buf_t serialize_hardcore(const LinearRegression &lr) {
    mousika::Buf_t buf;
    mousika::Marshal::serialize_append(buf,lr.w);
    mousika::Marshal::serialize_append(buf,lr.bias);
    return buf;
  }

  static LinearRegression deserialize_hardcore(const mousika::Buf_t &buf) {
    LinearRegression lr;
    bool res = mousika::Marshal::deserialize(buf,lr.w);
    assert(res);
    auto nbuf = mousika::Marshal::forward(buf,0,sizeof(double));
    res = mousika::Marshal::deserialize(nbuf,lr.bias);
    assert(res);
    return lr;
  }
  double bias, w;
#if REPORT_TNUM
  uint64_t num_training_set;
#endif
};

#endif  // MODEL_H
