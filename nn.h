#include "esp_dsp.h"
#define matmul dspm_mult_f32_ae32

float x1[1024];
float x2[1024];
float m1[32][32];
float m2[32][128];
float m3[32][128];
float b1[128];

float w1[1024];
float w2[1024];
float w3[1024];


void avgpool(float *x, float *y, int numseg, int n) {
    const float zero = 0.0;

    for (int i = 0; i < n; i++)
    y[i] = zero;

    float norm = 1.0 / (float)numseg;
    
    for (int i = 0; i < numseg; i++)
    for (int j = 0; j < n; j++)
    y[j] += x[j + i * n] * norm;
}


void sigmoid(float *x, int n) {
    const float one = 1.0;

    for (int i = 0; i < n; i++)
    x[i] = one / (one + exp(-x[i]));
}


void sigmoidapprxabs(float *x, int n) {
    const float half = 0.5;
    const float one = 1.0;
    const float two = 2.0;

    for (int i = 0; i < n; i++)
    x[i] = (x[i] / (two * (one + abs(x[i])))) + half;
}


void sigmoidapprxlin(float *x, int n) {
    const float c1 = -4.0, c2 = -2.0, c3 = 2.0, c4 = 4.0;
    const float a1 = 0.05, a2 = 0.2, a3 = 0.05;
    const float b1 = 0.2, b2 = 0.5, b3 = 0.8;
    const float zero = 0.0, one = 1.0;

    for (int i = 0; i < n; i++) {
        if (x[i] <= c1)      x[i] = zero;
        else if (x[i] <= c2) x[i] = a1 * x[i] + b1;
        else if (x[i] <= c3) x[i] = a2 * x[i] + b2;
        else if (x[i] <= c4) x[i] = a3 * x[i] + b3;
        else                 x[i] = one;
    }
}


void relu(float *x, int n) {
    const float zero = 0.0;
    for (int i = 0; i < n; i++)
    if (x[i] < zero) x[i] = zero;
}


void leakyrelu(float *x, float negslope, int n) {
    const float zero = 0.0;
    for (int i = 0; i < n; i++)
    if (x[i] < zero) x[i] *= negslope;
}


void hardswish(float *x, int n) {
    const float c1 = -3.0, c2 = 3.0;
    const float zero = 0.0, t = 3.0, s = 1.0/6.0;

    for (int i = 0; i < n; i++) {
        if (x[i] <= c1)      x[i] = zero;
        else if (x[i] <= c2) x[i] *= ((x[i] + t) * s);
    }
}


void mul(float *x, float *y, float *z, int n) {
    for (int i = 0; i < n; i++)
    z[i] = x[i] * y[i];
}


void transpose(float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
    y[j * rows + i] = x[i * cols + j];
}


void transpose(float *x, int dim) {
    for (int i = 0; i < dim; i++)
    for (int j = i + 1; j < dim; j++) {
        float temp = x[i * dim + j];
        x[i * dim + j] = x[j * dim + i];
        x[j * dim + i] = temp;
    }
}


void layernorm(float *x, float *y, int numseg, int m, float *gamma, float *beta) {
    const float eps = 0.00001;

    for (int seg = 0; seg < numseg; seg++) {
        int start_idx = seg * m;
        int end_idx = (seg + 1) * m;

        float mean = 0.0;
        for (int i = start_idx; i < end_idx; i++) mean += x[i];
        mean /= (float)m;

        float var = 0.0;
        float diff = 0.0;
        for (int i = start_idx; i < end_idx; i++) {
            diff = x[i] - mean;
            var += diff * diff;
        }
        var /= m;

        float stddev = sqrtf(var + eps);

        for (int i = start_idx; i < end_idx; i++)
        y[i] = ((x[i] - mean) / stddev) * gamma[i - start_idx] + beta[i - start_idx];
    }
}


void linear(float *x, float *w, float *y, int numseg, int in, int out) {
    matmul(x, w, y, numseg, in, out);
}


void linear(float *x, float *w, float *b, float *y, int numseg, int in, int out) {
    matmul(x, w, y, numseg, in, out);

    for (int seg = 0; seg < numseg; seg++) {
        int start_idx = seg * out;
        int end_idx = (seg + 1) * out;

        for (int i = start_idx; i < end_idx; i++)
        y[i] += b[i - start_idx];
    }
}


void conv2d1(float *x, float *w, float *b, float *y, int h, int w, int c, int k, int p) {    // do testów
  int out_h = h - k + 2*p + 1;
  int out_w = w - k + 2*p + 1;
  for (int i = 0; i < out_h; i++) {
    for (int j = 0; j < out_w; j++) {
      for (int l = 0; l < c; l++) {
        float patch[k*k];
        for (int m = 0; m < k; m++) {
          for (int n = 0; n < k; n++) {
            int x_i = i - p + m;
            int x_j = j - p + n;
            if (x_i >= 0 && x_i < h && x_j >= 0 && x_j < w) {
              patch[m*k + n] = x[(x_i*w + x_j)*c + l];
            } else {
              patch[m*k + n] = 0.0f;
            }
          }
        }
        float result;
        matmul(patch, w, &result, 1, k*k, 1);
        y[(i*out_w + j)*c + l] = result + b[l];
      }
    }
  }
}


void im2col(float* im, int h, int w, int c, int k, int p, float* col) {    // do testów
  int out_h = h - k + 2*p + 1;
  int out_w = w - k + 2*p + 1;
  for (int i = 0; i < out_h; i++) {
    for (int j = 0; j < out_w; j++) {
      for (int l = 0; l < c; l++) {
        for (int m = 0; m < k; m++) {
          for (int n = 0; n < k; n++) {
            int x_i = i - p + m;
            int x_j = j - p + n;
            if (x_i >= 0 && x_i < h && x_j >= 0 && x_j < w) {
              col[((i * out_w + j) * c + l) * k * k + m * k + n] = im[(x_i * w + x_j) * c + l];
            } else {
              col[((i * out_w + j) * c + l) * k * k + m * k + n] = 0.0f;
            }
          }
        }
      }
    }
  }
}

void conv2d2(float *x, float *w, float *b, float *y, int h, int w, int c, int k, int p) {    // do testów
  int out_h = h - k + 2*p + 1;
  int out_w = w - k + 2*p + 1;
  float col[out_h * out_w * c * k * k];
  im2col(x, h, w, c, k, p, col);
  matmul(col, w, y, out_h * out_w * c, k * k, 1);
  for (int i = 0; i < out_h * out_w * c; i++) {
    y[i] += b[i % c];
  }
}


void add(float *y, float *x, int n) {
    for (int i = 0; i < n; i++)
    y[i] += x[i];
}


void setup() {
    Serial.begin(115200);
    delay(1000);

    for (int i = 0; i < 1024; i++) {
        x1[i] = 0.012 * ((float)i - 512.0);
        x2[i] = 0.021 * ((float)i - 512.0);
    }

    int l = 0;
    for (int i = 0; i < 32; i++)
    for (int j = 0; j < 32; j++){
        m1[i][j] = 0.321 * ((float)l - 512.0);
        l++;
    }


    for (int i = 0; i < 32; i++)
    for (int j = 0; j < 128; j++)
    m2[i][j] = 0.123 * ((float)j - 64.0);



    unsigned int start_a = dsp_get_cpu_cycle_count();

    linear(*m1, *m2, *m3, 32, 32, 128);

    unsigned int end_a = dsp_get_cpu_cycle_count();


    unsigned int start_b = dsp_get_cpu_cycle_count();

    transpose(*m1, *m3, 32, 32);

    unsigned int end_b = dsp_get_cpu_cycle_count();


    unsigned int start_c = dsp_get_cpu_cycle_count();

    transpose(*m1, 32);

    unsigned int end_c = dsp_get_cpu_cycle_count();

    Serial.println();
    Serial.println();
    Serial.print("cykli a: ");
    Serial.println(end_a - start_a);
    Serial.print("cykli b: ");
    Serial.println(end_b - start_b);
    Serial.print("cykli c: ");
    Serial.println(end_c - start_c);
    Serial.println();
    Serial.println(w1[123], 3);
    Serial.println(w2[123], 3);
    Serial.println(m1[12][12], 3);
    Serial.println(m3[12][12], 3);

    Serial.println();
    Serial.println("Tablica m1:");
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        Serial.print(m1[i][j], 2); // Wyświetlenie wartości z dokładnością do 4 miejsc po przecinku
        Serial.print("\t");
      }
      Serial.println(); // Nowa linia po każdym wierszu
    }

    Serial.println("\nTablica m3:");
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < 32; j++) {
        Serial.print(m3[i][j], 2); // Wyświetlenie wartości z dokładnością do 4 miejsc po przecinku
        Serial.print("\t");
      }
      Serial.println(); // Nowa linia po każdym wierszu
    }
}


void loop() {}
