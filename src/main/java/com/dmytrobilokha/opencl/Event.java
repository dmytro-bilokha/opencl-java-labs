package com.dmytrobilokha.opencl;

import java.lang.foreign.MemorySegment;

public class Event {

    private final MemorySegment eventMemSeg;

    public Event(MemorySegment eventMemSeg) {
        this.eventMemSeg = eventMemSeg;
    }

    MemorySegment getEventMemSeg() {
        return eventMemSeg;
    }

}
