void wcumsum(const double *in_x,
                 const double *in_w,
                 unsigned int x_count,
                 unsigned int M,
                 double *out_sum) {
    unsigned int i, m;
    for (m=0;m<M;m++){
        out_sum[0] = in_x[0];
        for (i=1;i<x_count;i++){
            out_sum[i] = out_sum[i-1]*in_w[i] + in_x[i];
        }
        out_sum += x_count;
        in_w += x_count;
    }
}
