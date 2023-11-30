#include <sycl/sycl.hpp>
#include <random>
#include <sys/time.h>
using namespace sycl;

float sum_gpu;
float sum_cpu;
void gpu_cpu(size_t N){
    
    queue q(gpu_selector_v);

    std::vector<float> matrix_a(N*N);
    std::vector<float> matrix_b(N*N);
    std::vector<float> matrix_c(N*N);
    std::vector<float> matrix_d(N*N);

    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            matrix_a[i*N+j] = rand()%100;
            matrix_b[i*N+j] = rand()%100;
    }
    
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);
    
     struct timeval start;
     struct timeval end;
     gettimeofday (&start, NULL);
    
       q.submit([&](handler &h){
           
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

         range<2> global_size(N,N);
         range<2> work_group_size(10,10);

        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){        
             int i = item.get_local_id(0);
             int j = item.get_local_id(1);
             int i1 = item.get_group(0);
             int j1 = item.get_group(1);
            i+=10*i1;
            j+=10*j1;

            float temp = 0;
            for (int k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        });
    }).wait();
    
    host_accessor res(c, read_only);
    
    gettimeofday (&end, NULL);
    sum_gpu+=((end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0);
    
    
     struct timeval start1;
     struct timeval end1;
     gettimeofday (&start1, NULL);
    
     for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                matrix_d[i*N+j] += matrix_a[i*N+k] * matrix_b[k*N+j];
            }
            if(matrix_d[i*N+j] != res[i*N+j]){
                std::cout<<"结果程序在("<<i<<","<<"j"<<")位置上出错，程序退出";
                return ;
            }
        }
    }
    
    gettimeofday (&end1, NULL);
    sum_cpu+=((end1.tv_sec-start1.tv_sec)*1000+(end1.tv_usec-start1.tv_usec)/1000.0);
    
}


int  main(int argc, char**argv) {
    
    size_t N = atoi(argv[1]);
    std::cout<<"size:"<<N<<"*"<<N<<"\n";

    for(int c=0;c<1000;c++) gpu_cpu(N);
    
    std::cout<<"gpu: "<<sum_gpu/1000<<"ms"<<std::endl;
    std::cout<<"cpu: "<<sum_cpu/1000<<"ms"<<std::endl;
    
    return 0;
}

