//
// Created by jianan yuan on 2023/4/12.
//

#ifndef LEADER_PLR_H
#define LEADER_PLR_H

#include <string>
#include <vector>

// Code modified from https://github.com/RyanMarcus/plr

struct point {
    double x_;
    double y_;

    point() = default;
    point(double x, double y) : x_(x), y_(y) { }
};

struct line {
    double a_;
    double b_;
};

class Segment {
public:
    explicit Segment(double x=0, double k=0, double b=0, double x2=0) :
    x_(x), k_(k), b_(b), x2_(x2) { }
    double x_;
    double k_;
    double b_;
    double x2_;
};

double get_slope(const struct point& p1, const struct point& p2) {
  return (p2.y_ - p1.y_) / (p2.x_ - p1.x_);
}

struct line get_line(const struct point& p1, const struct point& p2) {
  double a = get_slope(p1, p2);
  double b = -a * p1.x_ + p1.y_;
  struct line l{.a_ = a, .b_ = b};
  return std::move(l);
}

struct point get_intersetction(const struct line& l1, const struct line& l2) {
  double a = l1.a_;
  double b = l2.a_;
  double c = l1.b_;
  double d = l2.b_;
  struct point p {(d - c) / (a - b), (a * d - b * c) / (a - b)};
  return std::move(p);
}

bool is_above(const struct point& pt, const struct line& l) {
  return pt.y_ > l.a_ * pt.x_ + l.b_;
}

bool is_below(const struct point& pt, const struct line& l) {
  return pt.y_ < l.a_ * pt.x_ + l.b_;
}

struct point get_upper_bound(const struct point& pt, double gamma) {
  struct point p {pt.x_, pt.y_ + gamma};
  return std::move(p);
}

struct point get_lower_bound(const struct point& pt, double gamma) {
  struct point p {pt.x_, pt.y_ - gamma};
  return std::move(p);
}

class GreedyPLR {
private:
    uint8_t state;
    double gamma;
    struct point last_pt;
    struct point s0;
    struct point s1;
    struct line rho_lower;
    struct line rho_upper;
    struct point sint;

    void setup() {
      this->rho_lower = std::move(get_line(get_upper_bound(this->s0, this->gamma),
                                           get_lower_bound(this->s1, this->gamma)));
      this->rho_upper = std::move(get_line(get_lower_bound(this->s0, this->gamma),
                                           get_upper_bound(this->s1, this->gamma)));
      this->sint = std::move(get_intersetction(this->rho_upper, this->rho_lower));
    }

    Segment current_segment() const {
      double segment_start = this->s0.x_;
      double avg_slope = (this->rho_lower.a_ + this->rho_upper.a_) / 2.0;
      double intercept = -avg_slope * this->sint.x_ + this->sint.y_;
      Segment s(segment_start, avg_slope, intercept, last_pt.x_);
      return std::move(s);
    }

    Segment process__(const struct point& pt) {
      if (!(is_above(pt, this->rho_lower) && is_below(pt, this->rho_upper))) {
        Segment prev_segment = std::move(current_segment());
        this->s0 = pt;
        //    this->state = "need1";
        this->state = 1;
        return std::move(prev_segment);
      }
      struct point s_upper = std::move(get_upper_bound(pt, this->gamma));
      struct point s_lower = std::move(get_lower_bound(pt, this->gamma));
      if (is_below(s_upper, this->rho_upper)) {
        this->rho_upper = std::move(get_line(this->sint, s_upper));
      }
      if (is_above(s_lower, this->rho_lower)) {
        this->rho_lower = std::move(get_line(this->sint, s_lower));
      }
      Segment s(0, 0, 0, 0);
      return std::move(s);
    }

public:
    explicit GreedyPLR(double gamma) {
      //  this->state = "need2";
      this->state = 2;
      this->gamma = gamma;
    }

    Segment process(const struct point& pt) {
      this->last_pt = pt;
      if (this->state == 2) {
        this->s0 = pt;
        this->state = 1;
      } else if (this->state == 1) {
        this->s1 = pt;
        setup();
        //    this->state = "ready";
        this->state = 3;
      } else if (this->state == 3) {
        return std::move(process__(pt));
      } else {
        // impossible
        std::cout << "ERROR in process" << std::endl;
      }
      Segment s(0, 0, 0, 0);
      return std::move(s);
    }

    Segment finish() {
      Segment s(0, 0, 0, 0);
      if (this->state == 2) {
        //    this->state = "finished";
        this->state = 4;
        return std::move(s);
      } else if (this->state == 1) {
        //    this->state = "finished";
        this->state = 4;
        s.x_ = this->s0.x_;
        s.k_ = 0;
        s.b_ = this->s0.y_;
        s.x2_ = this->last_pt.x_;
        return std::move(s);
      } else if (this->state == 3) {  // this->state = "ready"
        //    this->state = "finished";
        this->state = 4;
        return std::move(current_segment());
      } else {
        std::cout << "ERROR in finish" << std::endl;
        return std::move(s);
      }
    }
};

class PLR {
private:
    double gamma;
    std::vector<Segment> segments;

public:
    explicit PLR(double gamma) {
      this->gamma = gamma;
    }

    std::vector<Segment>& train(std::vector<double>& keys) {
      GreedyPLR plr_(this->gamma);
      size_t size = keys.size();
      segments.reserve(size + 1);
      for (int i = 0; i < size; ++i) {
        Segment seg = std::move(plr_.process(point(keys[i], i)));
        if (seg.x_ != 0 || seg.k_ != 0 || seg.b_ != 0) {
          segments.emplace_back(seg);
        }
      }
      Segment last = std::move(plr_.finish());
      if (last.x_ != 0 || last.k_ != 0 || last.b_ != 0) {
        segments.emplace_back(last);
      }
      segments.shrink_to_fit();
      return segments;
    }
};

#endif