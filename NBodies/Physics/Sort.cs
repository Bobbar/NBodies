using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public static class Sort
    {
        private static int _threshold = 5000; // Length at which to use .NET Array.Sort instead of QuickSort recurse/invoke.
        private static int _processorCount = Environment.ProcessorCount - 3; // = # cores/threads - 3 threads. (2 loop tasks, 1 UI thread)

        static void Swap(long[] keys, SpatialInfo[] data, int i, int j)
        {
            long temp = keys[i];
            keys[i] = keys[j];
            keys[j] = temp;

            SpatialInfo temp2 = data[i];
            data[i] = data[j];
            data[j] = temp2;
        }

        static int Partition(long[] keys, SpatialInfo[] data, int from, int to, int pivot)
        {
            var arrayPivot = keys[pivot];  // pivot value
            Swap(keys, data, pivot, to - 1); // move pivot value to end for now, after this pivot not used
            var newPivot = from; // new pivot 
            for (int i = from; i < to - 1; i++) // be careful to leave pivot value at the end
            {
                if (keys[i] < arrayPivot)
                {
                    Swap(keys, data, newPivot, i);  // move value smaller than arrayPivot down to newpivot
                    newPivot++;
                }
            }
            Swap(keys, data, newPivot, to - 1); // move pivot value to its final place
            return newPivot; // new pivot
        }

        public static void ParallelQuickSort(long[] keys, SpatialInfo[] data, int length)
        {
            ParallelQuickSort(keys, data, 0, length, _processorCount);
        }
       
        static void ParallelQuickSort(long[] keys, SpatialInfo[] data, int from, int to, int depthRemaining)
        {
            if (to - from <= _threshold)
            {
                Array.Sort(keys, data, from, to - from);
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