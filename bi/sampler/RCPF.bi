function quantile(samples:Real[_], p:Real) -> Real {
  assert 0 < p && p < 1;
  auto ssamples <- sort(samples);
  auto N <- length(ssamples);
  auto i <- Integer(1+floor(N*p));
  return ssamples[i];
}

class RCPF < AliveParticleFilter {
  thresholdQuantile:Real <- 0.8;
  zeroThresholdMode:String <- "apf";

  C:Queue<Real>;

  /*
  function start() {
    for auto n in 1..N {
      x[n].h.setMode(PLAY_IMMEDIATE);
    }
    super.start();
  }
  */

  function step() {
    cpp {{
    libbirch::Atomic<bi::type::Integer> P(0);
    }}
    auto x0 <- x;
    auto w0 <- w;
    parallel for auto n in 1..N {
      x[n] <- clone<ForwardModel>(x0[a[n]]);
      w[n] <- x[n].step();
      cpp {{
      ++P;
      }}
    }

    auto c <- quantile(w, thresholdQuantile);
    C.pushBack(c);

    acc:Boolean[N];
    parallel for auto n in 1..N {
      if c == -inf {
        if zeroThresholdMode == "apf" {
          acc[n] <- (w[n] != -inf);
        } else {
          acc[n] <- true;
        }
      } else if w[n] >= c {
        acc[n] <- true;
      } else {
        acc[n] <- simulate_bernoulli(exp(w[n] - c));
        w[n] <- c;
      }
      while !acc[n] {
        a[n] <- ancestor(w0);
        x[n] <- clone<ForwardModel>(x0[a[n]]);
        w[n] <- x[n].step();
        cpp {{
        ++P;
        }}
        if c == -inf {
          if zeroThresholdMode == "apf" {
            acc[n] <- (w[n] != -inf);
          } else {
            acc[n] <- true;
          }
        } else if w[n] >= c {
          acc[n] <- true;
        } else {
          acc[n] <- simulate_bernoulli(exp(w[n] - c));
          w[n] <- c;  // if not accepted, it will be overwritten anyway
        }
      }
    }

    /* propagate and weight until one further acceptance, which is discarded
     * for unbiasedness in the normalizing constant estimate */
    w':Real;
    auto acc' <- false;
    do {
      auto a' <- ancestor(w0);
      auto x' <- clone<ForwardModel>(x0[a']);
      w' <- x'.step();
      cpp {{
      ++P;
      }}
      if c == -inf {
        if zeroThresholdMode == "apf" {
          acc' <- (w' != -inf);
        } else {
          acc' <- true;
        }
      } else if w' >= c {
        acc' <- true;
      } else {
        acc' <- simulate_bernoulli(exp(w' - c));
      }
    } while !acc';

    /* update propagations */
    Q:Integer;
    cpp{{
    Q = P.load();
    }}
    this.P.pushBack(Q);
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    thresholdQuantile <-? buffer.get("threshold_quantile", thresholdQuantile);
    zeroThresholdMode <-? buffer.get("zero_threshold_mode", zeroThresholdMode);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("threshold_quantile", thresholdQuantile);
    buffer.set("zero_threshold_mode", zeroThresholdMode);
    buffer.set("thresholds", C);
  }
}
