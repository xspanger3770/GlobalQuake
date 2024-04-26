package globalquake.core.database;

import javax.annotation.Nonnull;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

public class CountInputStream extends FilterInputStream {

    private long count = 0L;
    private Runnable event;

    public CountInputStream(InputStream in) {
        super(in);
    }

    public void setEvent(Runnable event) {
        this.event = event;
    }

    public int read() throws IOException {
        final int c = super.read();
        if (c >= 0) {
            count++;
            event.run();
        }
        return c;
    }

    public int read(@Nonnull byte[] b, int off, int len) throws IOException {
        final int bytesRead = super.read(b, off, len);
        if (bytesRead > 0) {
            count += bytesRead;
            event.run();
        }
        return bytesRead;
    }

    public int read(@Nonnull byte[] b) throws IOException {
        final int bytesRead = super.read(b);
        if (bytesRead > 0) {
            count += bytesRead;
            event.run();
        }
        return bytesRead;
    }

    public long getCount() {
        return count;
    }
}
