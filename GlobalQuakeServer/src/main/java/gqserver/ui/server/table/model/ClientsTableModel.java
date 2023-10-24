package gqserver.ui.server.table.model;

import gqserver.api.ServerClient;
import gqserver.ui.server.table.Column;
import gqserver.ui.server.table.LastUpdateRenderer;
import gqserver.ui.server.table.TableCellRendererAdapter;

import java.time.LocalDateTime;
import java.util.List;

public class ClientsTableModel extends FilterableTableModel<ServerClient>{
    private final List<Column<ServerClient, ?>> columns = List.of(
            Column.readonly("ID", Integer.class, ServerClient::getID, new TableCellRendererAdapter<>()),
            Column.readonly("Joined at", LocalDateTime.class, ServerClient::getJoinDate, new LastUpdateRenderer<>()),
            Column.readonly("Delay (ms)", Long.class, ServerClient::getDelay, new TableCellRendererAdapter<>()),
            Column.readonly("Packets sent", Long.class, ServerClient::getSentPackets, new TableCellRendererAdapter<>()),
            Column.readonly("Packets received", Long.class, ServerClient::getReceivedPackets, new TableCellRendererAdapter<>()));


    public ClientsTableModel(List<ServerClient> data) {
        super(data);
    }

    @Override
    public int getColumnCount() {
        return columns.size();
    }

    @Override
    public String getColumnName(int columnIndex) {
        return columns.get(columnIndex).getName();
    }

    public TableCellRendererAdapter<?, ?> getColumnRenderer(int columnIndex) {
        return columns.get(columnIndex).getRenderer();
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        return columns.get(columnIndex).getColumnType();
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columns.get(columnIndex).isEditable();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        ServerClient event = getEntity(rowIndex);
        return columns.get(columnIndex).getValue(event);
    }

    @Override
    public void setValueAt(Object value, int rowIndex, int columnIndex) {
        ServerClient event = getEntity(rowIndex);
        columns.get(columnIndex).setValue(value, event);
    }
}
