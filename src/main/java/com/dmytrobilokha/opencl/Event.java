package com.dmytrobilokha.opencl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class Event {

    private final MemorySegment eventMemSeg;

    private Event(MemorySegment eventMemSeg) {
        this.eventMemSeg = eventMemSeg;
    }

    public static Event fromPointer(MemorySegment eventPointerMemSeg) {
        return new Event(eventPointerMemSeg.get(ValueLayout.ADDRESS, 0));
    }

    MemorySegment getEventMemSeg() {
        return eventMemSeg;
    }

}
