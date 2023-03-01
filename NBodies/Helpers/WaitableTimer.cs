using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace NBodies.Helpers
{
	public class WaitableTimer : IDisposable
	{
		private IntPtr handle;
		private bool disposedValue;
		private const uint INFINITE_TIMEOUT = 0xFFFFFFFF;

		public WaitableTimer() : this(IntPtr.Zero, null, CreateFlags.CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, AccessFlags.TIMER_ALL_ACCESS)
		{ }

		public WaitableTimer(IntPtr attributes, string name, IntPtr flags, IntPtr access)
		{
			var handle = CreateWaitableTimerExW(attributes, name, flags, access);

			if (handle == IntPtr.Zero)
			{
				throw new Exception("Failed to create timer.");
			}
			else
			{
				this.handle = handle;
			}
		}

		public void Wait(long dueTime, bool resume)
		{
			var dt = dueTime * -1;
			SetWaitableTimer(handle, ref dt, 0, null, IntPtr.Zero, resume);
			WaitForSingleObject(handle, INFINITE_TIMEOUT);
		}

		protected virtual void Dispose(bool disposing)
		{
			if (!disposedValue)
			{
				if (disposing)
				{
					// TODO: dispose managed state (managed objects)
				}

				// TODO: free unmanaged resources (unmanaged objects) and override finalizer
				// TODO: set large fields to null

				var res = CloseHandle(handle);

				if (res)
					this.handle = IntPtr.Zero;

				disposedValue = true;
			}
		}

		// // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
		// ~WaitableTimer()
		// {
		//     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
		//     Dispose(disposing: false);
		// }

		public void Dispose()
		{
			// Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
			Dispose(disposing: true);
			GC.SuppressFinalize(this);
		}


		public static class CreateFlags
		{
			public static readonly IntPtr CREATE_WAITABLE_TIMER_MANUAL_RESET = new IntPtr(0x00000001);
			public static readonly IntPtr CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = new IntPtr(0x00000002);
		}

		public static class AccessFlags
		{
			public static readonly IntPtr TIMER_ALL_ACCESS = new IntPtr(0x1F0003);
			public static readonly IntPtr TIMER_MODIFY_STATE = new IntPtr(0x0002);
			public static readonly IntPtr TIMER_QUERY_STATE = new IntPtr(0x0001);
		}

		[DllImport("kernel32", SetLastError = true, ExactSpelling = true)]
		public static extern IntPtr CreateWaitableTimerExW(IntPtr lpTimerAttributes, string lpTimerName, IntPtr dwFlags, IntPtr dwDesiredAccess);

		[DllImport("kernel32", SetLastError = true, ExactSpelling = true)]
		public static extern bool CloseHandle(IntPtr hObject);

		public delegate void TimerAPCProc(
		   IntPtr lpArgToCompletionRoutine,
		   UInt32 dwTimerLowValue,
		   UInt32 dwTimerHighValue);

		[DllImport("kernel32", SetLastError = true, ExactSpelling = true)]
		public static extern bool SetWaitableTimer(IntPtr hTimer, [In] ref long pDueTime, Int64 lPeriod, TimerAPCProc pfnCompletionRoutine, IntPtr lpArgToCompletionRoutine, bool fResume);

		[DllImport("kernel32", SetLastError = true, ExactSpelling = true)]
		public static extern Int32 WaitForSingleObject(IntPtr handle, uint milliseconds);


		public struct LARGE_INTEGER
		{
			struct DUMMY
			{
				UInt32 LowPart;
				Int64 HighPart;
			}

			struct u
			{
				UInt32 LowPart;
				Int64 HighPart;
			}

			public Int64 QuadPart;
		}


	}
}
