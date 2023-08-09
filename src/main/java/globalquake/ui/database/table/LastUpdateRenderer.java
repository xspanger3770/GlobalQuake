package globalquake.ui.database.table;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class LastUpdateRenderer<E> extends TableCellRendererAdapter<E, LocalDateTime> {

	private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm:ss");

	@SuppressWarnings("unused")
	@Override
	public String getText(E entity, LocalDateTime value) {
		if(value == null){
			return "Never";
		}
		return formatter.format(value);
	}

}
