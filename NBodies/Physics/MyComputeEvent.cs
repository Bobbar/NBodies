using Cloo;
using Cloo.Bindings;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;


namespace NBodies.Physics
{
    public class MyComputeEvent : ComputeEventBase
    {
        /// <summary>
        /// Gets the <see cref="ComputeCommandQueue"/> associated with the <see cref="ComputeEvent"/>.
        /// </summary>
        /// <value> The <see cref="ComputeCommandQueue"/> associated with the <see cref="ComputeEvent"/>. </value>
        public ComputeCommandQueue CommandQueue { get; }


        public MyComputeEvent(CLEventHandle handle, ComputeCommandQueue queue) : this(handle, queue, 0)
        {
            Type = (ComputeCommandType)GetInfo<CLEventHandle, ComputeEventInfo, int>(Handle, ComputeEventInfo.CommandType, CL12.GetEventInfo);
        }

        public MyComputeEvent(CLEventHandle handle, ComputeCommandQueue queue, ComputeCommandType type)
        {
            Handle = handle;
            SetID(Handle.Value);

            CommandQueue = queue;
            Type = type;
            Context = queue.Context;
        }

        internal void TrackGCHandle(GCHandle gcHandle)
        {
            var freeDelegate = new ComputeCommandStatusChanged((s, e) =>
            {
                if (gcHandle.IsAllocated && gcHandle.Target != null) gcHandle.Free();
            });

            Completed += freeDelegate;
            Aborted += freeDelegate;
        }

        public override ComputeEventBase Clone()
        {
            CL10.RetainEvent(Handle);
            return new MyComputeEvent(Handle, CommandQueue, Type);
        }
    }
}
