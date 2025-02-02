package com.dmytrobilokha;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class FileUtil {

    private FileUtil() {
        // no instance
    }

    public static String readStringResource(String resourcePath) {
        try (InputStream csvStream = FileUtil.class.getModule().getResourceAsStream(resourcePath)) {
            return new String(csvStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Failed to read resource: " + resourcePath, e);
        }
    }

}
