using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public static class Sort
    {
        public static int Threshold = 20000; // keys length to use InsertionSort instead of SequentialQuickSort
        public static int ProcessorCount = 5;

        public static unsafe void InsertionSort(int* keys, SpatialInfo* data, int from, int to)
        {
            for (int i = from + 1; i < to; i++)
            {
                int a = keys[i];
                SpatialInfo ad = data[i];
                int j = i - 1;

                while (j >= from && a > keys[j])
                {
                    keys[j + 1] = keys[j];
                    data[j + 1] = data[j];
                    j--;
                }
                keys[j + 1] = a;
                data[j + 1] = ad;
            }
        }

        static unsafe void SwapPtr(int* keys, SpatialInfo* data, int i, int j)
        {
            int temp = keys[i];
            keys[i] = keys[j];
            keys[j] = temp;

            SpatialInfo temp2 = data[i];
            data[i] = data[j];
            data[j] = temp2;
        }

        static unsafe int Partition(int* keys, SpatialInfo* data, int from, int to, int pivot)
        {
            int arrayPivot = keys[pivot];  // pivot value
            SwapPtr(keys, data, pivot, to - 1); // move pivot value to end for now, after this pivot not used
            int newPivot = from; // new pivot 
            for (int i = from; i < to - 1; i++) // be careful to leave pivot value at the end
            {
                if (keys[i] > arrayPivot)
                {
                    SwapPtr(keys, data, newPivot, i);  // move value smaller than arrayPivot down to newpivot
                    newPivot++;
                }
            }
            SwapPtr(keys, data, newPivot, to - 1); // move pivot value to its final place
            return newPivot; // new pivot
        }

        public static unsafe void ParallelQuickSort(int* keys, SpatialInfo* data, int length)
        {
            ParallelQuickSort(keys, data, 0, length, ProcessorCount);
        }

        static unsafe void ParallelQuickSort(int* keys, SpatialInfo* data, int from, int to, int depthRemaining)
        {
            if (to - from <= Threshold)
            {
                InsertionSort(keys, data, from, to);
            }
            else
            {
                int pivot = from + ((to - from) >> 1);
                pivot = Partition(keys, data, from, to, pivot);
                if (depthRemaining > 0)
                {
                    Parallel.Invoke(
                        () => ParallelQuickSort(keys, data, from, pivot, depthRemaining - 1),
                        () => ParallelQuickSort(keys, data, pivot + 1, to, depthRemaining - 1));
                }
                else
                {
                    ParallelQuickSort(keys, data, from, pivot, 0);
                    ParallelQuickSort(keys, data, pivot + 1, to, 0);
                }
            }
        }
    }
}
