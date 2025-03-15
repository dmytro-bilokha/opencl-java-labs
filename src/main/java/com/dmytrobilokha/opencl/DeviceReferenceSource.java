package com.dmytrobilokha.opencl;

import java.util.Map;
import java.util.regex.Pattern;

// TODO: add more data here
public class DeviceReferenceSource {

    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");

    private final Map<String, Long> deviceNumberOfCoresMap
            = Map.of(toKey("NVIDIA GeForce GT 1030"), 384L);

    private static String toKey(String deviceName) {
        return WHITESPACE_PATTERN.matcher(deviceName).replaceAll("").toUpperCase();
    }

    public Long getNumberOfCores(String deviceName) {
        String deviceKey = toKey(deviceName);
        Long propertyOverride = PropertyUtil.getAsLong(deviceKey + ".number.of.cores");
        if (propertyOverride != null) {
            return propertyOverride;
        }
        return deviceNumberOfCoresMap.get(deviceKey);
    }

}
