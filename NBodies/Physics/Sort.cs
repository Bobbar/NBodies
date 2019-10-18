using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NBodies.Physics
{
    public static class Sort
    {
        public static int Threshold = 150; // keys length to use InsertionSort instead of SequentialQuickSort

        public static void InsertionSort(int[] keys, SpatialInfo[] data,  int from, int to)
        {
            for (int i = from + 1; i < to; i++)
            {
                var a = keys[i];
                var ad = data[i];
                int j = i - 1;

                //while (j >= from && keys[j] > a)
                //while (j >= from && keys[j].CompareTo(a) == -1)
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

        static void Swap(int[] keys, SpatialInfo[] data, int i, int j)
        {
            int temp = keys[i];
            keys[i] = keys[j];
            keys[j] = temp;

            SpatialInfo temp2 = data[i];
            data[i] = data[j];
            data[j] = temp2;
        }

        static int Partition(int[] keys, SpatialInfo[] data, int from, int to, int pivot)
        {
            // Pre: from <= pivot < to (other than that, pivot is arbitrary)
            var arrayPivot = keys[pivot];  // pivot value
            Swap(keys, data, pivot, to - 1); // move pivot value to end for now, after this pivot not used
            var newPivot = from; // new pivot 
            for (int i = from; i < to - 1; i++) // be careful to leave pivot value at the end
            {
                // Invariant: from <= newpivot <= i < to - 1 && 
                // forall from <= j <= newpivot, keys[j] <= arrayPivot && forall newpivot < j <= i, keys[j] > arrayPivot
                // if (keys[i].CompareTo(arrayPivot) != -1)
                if (keys[i] > arrayPivot)
                {
                    Swap(keys, data, newPivot, i);  // move value smaller than arrayPivot down to newpivot
                    newPivot++;
                }
            }
            Swap(keys, data, newPivot, to - 1); // move pivot value to its final place
            return newPivot; // new pivot
            // Post: forall i <= newpivot, keys[i] <= keys[newpivot]  && forall i > ...
        }

        public static void SequentialQuickSort(int[] keys, SpatialInfo[] data)
        {
            SequentialQuickSort(keys, data, 0, keys.Length);
        }

        static void SequentialQuickSort(int[] keys, SpatialInfo[] data, int from, int to)
        {
            if (to - from <= Threshold)
            {
                InsertionSort(keys, data, from, to);
            }
            else
            {
                int pivot = from + ((to - from) >> 1); // could be anything, use middle
                pivot = Partition(keys, data, from, to, pivot);
                // Assert: forall i < pivot, keys[i] <= keys[pivot]  && forall i > ...
                SequentialQuickSort(keys, data, from, pivot);
                SequentialQuickSort(keys, data, pivot + 1, to);
            }
        }

        public static void ParallelQuickSort(int[] keys, SpatialInfo[] data, int length)
        {
            ParallelQuickSort(keys, data, 0, length,
                 (int)Math.Log(Environment.ProcessorCount, 2) + 4);
        }

        public static void ParallelQuickSort(int[] keys, SpatialInfo[] data)
        {
            ParallelQuickSort(keys, data, 0, keys.Length,
                 (int)Math.Log(Environment.ProcessorCount, 2) + 4);
        }

        static void ParallelQuickSort(int[] keys, SpatialInfo[] data, int from, int to, int depthRemaining)
        {
            if (to - from <= Threshold)
            {
                InsertionSort(keys, data, from, to);
            }
            else
            {
                //int pivot = from + (to - from) / 2; // could be anything, use middle
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
