#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define INF INT_MAX

using namespace std;

__global__ void ssspGPU(int V, const int *rowPtr, const int *col,
                        const int *weight, int *dist, bool *changed){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V)return;
    int dv = dist[tid];
    if (dv == INF)return;

    for (int i = rowPtr[tid]; i < rowPtr[tid + 1]; i++){
        int u = col[i];
        int w = weight[i];
        int newDist = dv + w;
        if (newDist < dist[u]){
            atomicMin(&dist[u], newDist);
            *changed = true;
        }
    }
}

void cpuSSSP(int V, const vector<int> &rowPtr,const vector<int> &col,
              const vector<int> &weight,vector<int> &dist){
    dist[0] = 0;
    for (int iter = 0; iter < V - 1; iter++){
        bool changed = false;
        for (int v = 0; v < V; v++){
            if (dist[v] == INF)continue;
            for (int i = rowPtr[v]; i < rowPtr[v + 1]; i++){
                int u = col[i];
                int newDist = dist[v] + weight[i];
                if (newDist < dist[u]){
                    dist[u] = newDist;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
}

int main(){
    ifstream file("graph.txt");
    if (!file){
        cout << "Could not open the file graph.txt\n";
        return 1;
    }
    int V, E;
    file >> V >> E;
    vector<int> src(E), dst(E), weight(E);
    vector<int> degree(V, 0);

    for (int i = 0; i < E; i++){
        file >> src[i] >> dst[i] >> weight[i];
        degree[src[i]]++;
    }

    vector<int> rowPtr(V + 1, 0);
    for (int i = 0; i < V; i++)rowPtr[i + 1] = rowPtr[i] + degree[i];

    vector<int> col(E);
    vector<int> csrWeight(E);
    vector<int> pos = rowPtr;

    for (int i = 0; i < E; i++){
        int v = src[i];
        col[pos[v]] = dst[i];
        csrWeight[pos[v]] = weight[i];
        pos[v]++;
    }
    vector<int> cpuDist(V, INF);
    cpuSSSP(V, rowPtr, col, csrWeight, cpuDist);
   
    cout << "CPU SSSP:\n";
    for (int i = 0; i < V; i++)
        cout << "0 -> " << i << " = " << cpuDist[i] << "\n";

    int *d_rowPtr, *d_col, *d_weight, *d_dist;
    bool *d_changed;
    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_weight, E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    cudaMemcpy(d_rowPtr, rowPtr.data(),(V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col.data(),E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, csrWeight.data(),E * sizeof(int), cudaMemcpyHostToDevice);
    vector<int> gpuDist(V, INF);
    gpuDist[0] = 0;
    cudaMemcpy(d_dist, gpuDist.data(),V * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (V + threads - 1) / threads;
    for (int i = 0; i < V - 1; i++){
        bool changed = false;
        cudaMemcpy(d_changed, &changed,sizeof(bool), cudaMemcpyHostToDevice);
        ssspGPU<<<blocks, threads>>>(V, d_rowPtr, d_col, d_weight, d_dist, d_changed);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, d_changed,sizeof(bool), cudaMemcpyDeviceToHost);

        if (!changed)break;
    }

    cudaMemcpy(gpuDist.data(), d_dist,V * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nGPU SSSP:\n";
    for (int i = 0; i < V; i++)
        cout << "0 -> " << i << " = " << gpuDist[i] << "\n";

    bool correct = true;
    for (int i = 0; i < V; i++){
        if (cpuDist[i] != gpuDist[i]){
            correct = false;
            break;
        }
    }

    cout << "\nResult: ";
    if (correct)
        cout << "CPU and GPU results are CORRECT and BOTH MATCH.\n";
    else
        cout << "CPU and GPU results DO NOT MATCH.\n";

    cudaFree(d_rowPtr);
    cudaFree(d_col);
    cudaFree(d_weight);
    cudaFree(d_dist);
    cudaFree(d_changed);

    return 0;
}