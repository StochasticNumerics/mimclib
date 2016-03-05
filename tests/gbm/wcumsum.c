void wcumsum(const double *in_x, const double *in_w,
             unsigned int in_count, double *out_sum) {
    unsigned int i;
    out_sum[0] = in_x[0];
    for (i=1;i<in_count;i++){
        out_sum[i] = out_sum[i-1]*in_w[i] + in_x[i];
    }
}
