package globalquake.ui.database.table;

import globalquake.ui.settings.Settings;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

public class LastUpdateRenderer<E> extends TableCellRendererAdapter<E, LocalDateTime> {

	@SuppressWarnings("unused")
	@Override
	public String getText(E entity, LocalDateTime value) {
		if(value == null){
			return "Never";
		}
		return Settings.selectedDateTimeFormat().format(value);
	}

}
