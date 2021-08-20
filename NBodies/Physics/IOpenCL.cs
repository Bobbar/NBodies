using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cloo;
using Cloo.Bindings;
using System.Runtime.InteropServices;
using AdvancedDLSupport;


namespace NBodies.Physics
{
    public static class OCTools
    {
        public static IntPtr[] ConvertArray(long[] array)
        {
            if (array == null) return null;

            IntPtr[] result = new IntPtr[array.Length];
            for (long i = 0; i < array.Length; i++)
                result[i] = new IntPtr(array[i]);
            return result;
        }
    }
    public static class MyOCLBinding
    {
        private static IOpenCL LibCalli = new NativeLibraryBuilder(ImplementationOptions.UseIndirectCalls).ActivateInterface<IOpenCL>("OpenCL");


        public static ComputeErrorCode EnqueueReadBuffer(CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            bool blocking_read,
            IntPtr offset,
            IntPtr cb,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
             CLEventHandle[] event_wait_list,
            CLEventHandle[] new_event)
        {
            return LibCalli.clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, new_event);
        }

        public static ComputeErrorCode EnqueueNDRangeKernel(CLCommandQueueHandle command_queue,
           CLKernelHandle kernel,
           Int32 work_dim,
           IntPtr[] global_work_offset,
            long[] global_work_size,
           long[] local_work_size,
           Int32 num_events_in_wait_list,
           CLEventHandle[] event_wait_list,
          CLEventHandle[] new_event)
        {
            return LibCalli.clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, new_event);
        }


        public static unsafe void SetMemoryArgument(CLKernelHandle kernel, int index, ComputeMemory memObj)
        {
            var sz = new IntPtr(Marshal.SizeOf(typeof(CLMemoryHandle)));

            LibCalli.clSetKernelArg(kernel.Value, (IntPtr)index, sz.ToPointer(), new IntPtr[] { memObj.Handle.Value });
        }
       
        public static unsafe void SetValueArgument<T>(CLKernelHandle kernel, int index, T data) where T : struct
        {
            var sz = new IntPtr(Marshal.SizeOf(typeof(T)));

            GCHandle gcHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                LibCalli.clSetKernelArg(kernel.Value, (IntPtr)index, sz.ToPointer(), new IntPtr[] { gcHandle.AddrOfPinnedObject() });

            }
            finally
            {
                gcHandle.Free();
            }
        }

      

        public static unsafe ComputeErrorCode SetKernelArg(
           IntPtr kernel,
           IntPtr arg_index,
           IntPtr arg_size,
           IntPtr arg_value)
        {
            return LibCalli.clSetKernelArg(kernel, arg_index, arg_size.ToPointer(), new IntPtr[] { arg_value });
        }

        //// Works.
        //public static unsafe ComputeErrorCode SetKernelArg(
        //    IntPtr kernel,
        //    IntPtr arg_index,
        //    IntPtr arg_size,
        //    CLMemoryHandle arg_value)
        //{
        //    //return LibCalli.clSetKernelArg(kernel, arg_index, arg_size, arg_value );
        //    //return LibCalli.clSetKernelArg(kernel, arg_index, arg_size.ToPointer(), new Span<CLMemoryHandle>(new CLMemoryHandle[] { arg_value }));

        //    return LibCalli.clSetKernelArg(kernel, arg_index, arg_size.ToPointer(), new CLMemoryHandle[] { arg_value });

        //}

        //public static ComputeErrorCode EnqueueNDRangeKernel(CLCommandQueueHandle command_queue,
        //    CLKernelHandle kernel,
        //    Int32 work_dim,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_offset,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_size,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] local_work_size,
        //    Int32 num_events_in_wait_list,
        //    [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
        //    [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event)
        //{
        //    return LibCalli.clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, new_event);
        //}

    }


    public interface IOpenCL
    {
        ComputeErrorCode clEnqueueReadBuffer(
            CLCommandQueueHandle command_queue,
            CLMemoryHandle buffer,
            bool blocking_read,
            IntPtr offset,
            IntPtr cb,
            IntPtr ptr,
            Int32 num_events_in_wait_list,
            CLEventHandle[] event_wait_list,
            CLEventHandle[] new_event);


        ComputeErrorCode clEnqueueNDRangeKernel(
           CLCommandQueueHandle command_queue,
           CLKernelHandle kernel,
           Int32 work_dim,
            IntPtr[] global_work_offset,
           long[] global_work_size,
           long[] local_work_size,
           Int32 num_events_in_wait_list,
            CLEventHandle[] event_wait_list,
            CLEventHandle[] new_event);

        // Works.
        unsafe ComputeErrorCode clSetKernelArg(
         IntPtr kernel,
         IntPtr arg_index,
         void* arg_size,
        IntPtr[] arg_value);

      
        
        
        
        
        //unsafe ComputeErrorCode clSetKernelArg(
        //   IntPtr kernel,
        //   IntPtr arg_index,
        //   void* arg_size,
        //  Span<CLMemoryHandle> arg_value);

        //ComputeErrorCode clSetKernelArg<T>(
        //    CLKernelHandle kernel,
        //    Int32 arg_index,
        //    IntPtr arg_size,
        //    T arg_value) where T : unmanaged;



        //ComputeErrorCode clEnqueueNDRangeKernel(
        //    CLCommandQueueHandle command_queue,
        //    CLKernelHandle kernel,
        //    Int32 work_dim,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_offset,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] global_work_size,
        //    [MarshalAs(UnmanagedType.LPArray)] IntPtr[] local_work_size,
        //    Int32 num_events_in_wait_list,
        //    [MarshalAs(UnmanagedType.LPArray)] CLEventHandle[] event_wait_list,
        //    [Out, MarshalAs(UnmanagedType.LPArray, SizeConst = 1)] CLEventHandle[] new_event);

    }
}
