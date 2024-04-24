package globalquake.ui.table;

import globalquake.core.Settings;

import java.time.LocalDateTime;

public class LastUpdateRenderer<E> extends TableCellRendererAdapter<E, LocalDateTime> {

    @SuppressWarnings("unused")
    @Override
    public String getText(E entity, LocalDateTime value) {
        if (value == null) {
            return "Never";
        }
        return Settings.formatDateTime(value);
    }

}
