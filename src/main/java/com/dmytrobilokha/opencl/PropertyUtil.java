package com.dmytrobilokha.opencl;

public class PropertyUtil {

    private PropertyUtil() {
        // no instance
    }

    public static Long getAsLong(String propertyKey) {
        var propertyString = System.getProperty(propertyKey);
        if (propertyString == null) {
            return null;
        }
        return Long.parseLong(propertyString);
    }

    public static String getAsString(String propertyKey, String defaultValue) {
        return System.getProperty(propertyKey, defaultValue);
    }

}
