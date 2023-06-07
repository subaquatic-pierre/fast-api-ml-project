import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { useNavigate, useParams } from 'react-router-dom'

import {
  CFormInput,
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CCardBody,
  CCard,
  CCardHeader,
} from '@coreui/react'
import CIcon from '@coreui/icons-react'

import CarImage from 'src/components/CarImage'
import CarEditImage from 'src/components/CarEditImage'

import { API_URL } from 'src/const'

const TableEditRow = ({ label, updateReport, car_index, label_index }) => {
  const [labelValues, setLabelValues] = useState(label)

  const handleInputChange = (e) => {
    setLabelValues((oldValues) => ({
      ...oldValues,
      [e.target.name]: e.target.value,
    }))
  }

  const handleConfirmClick = () => {
    updateReport(labelValues, car_index, label_index)
  }

  return (
    <CTableRow className="edit-row-cell" v-for="user in tableUsers">
      <CTableDataCell>{label.part}</CTableDataCell>
      <CTableDataCell>
        <CFormInput onChange={handleInputChange} value={labelValues.issue} name="issue" />
      </CTableDataCell>
      <CTableDataCell>
        <CFormInput
          onChange={handleInputChange}
          value={labelValues.repair_status}
          name="repair_status"
        />
      </CTableDataCell>
      <CTableDataCell>
        <CFormInput onChange={handleInputChange} value={labelValues.price} name="price" />
      </CTableDataCell>
      <CTableDataCell>
        <CFormInput onChange={handleInputChange} value={labelValues.labcharge} name="labcharge" />
      </CTableDataCell>
      <CTableDataCell className="text-end">
        <div>
          <button className="btn btn-success text-white" type="button" onClick={handleConfirmClick}>
            Confirm
          </button>
        </div>
      </CTableDataCell>
    </CTableRow>
  )
}

const defaultCase = {
  id: 'string',
  type: 'string',
  userId: 'string',
  createdDate: 'string',
  status: 'PROCESSED',
  reportGenerated: true,
  topViewGenerated: false,
  reportUrl: 'string',
  reportUrlPdf: 'string',
  reportUrlPdfTopView: 'string',
  vehicleCount: 0,
  report: [],
  all_final_dict: [],
}

const Case = () => {
  const [editRowIndex, setEditRowIndex] = useState('-1,-1')
  const [editCarIndex, setEditCarIndex] = useState('-1,-1')
  const params = useParams()
  const navigate = useNavigate()
  const [apiCase, setCase] = useState(defaultCase)

  const fetchCase = async () => {
    const url = `${API_URL}/case/${params.caseId}`
    try {
      const res = await axios.get(url)
      const resData = res.data
      const data = resData.data
      if (res.status === 200) {
        setCase(data)
      }
    } catch (e) {
      console.log(e)
    }
  }

  const handleEditRowClick = (car_index, label_index) => {
    setEditRowIndex(`${car_index},${label_index}`)
  }

  const handleUpdateReport = (newValues, car_index, label_index) => {
    const newCase = apiCase
    newCase.report[car_index].label_text[label_index] = newValues
    setCase(newCase)
    setEditRowIndex('-1,-1')
  }

  const handleAddRowClick = (car_index) => {
    console.log(car_index)
  }

  const handleCloseCarImageEditor = () => {
    setEditCarIndex('-1,-1')
  }

  const handleDeleteButtonClick = async () => {
    const url = `${API_URL}/case/${apiCase.id}`
    try {
      const res = await axios.delete(url)
      if (res.data === true) {
        navigate('/')
      }
    } catch (e) {
      console.log(e)
    }
  }

  useEffect(() => {
    fetchCase()
  }, [])

  return (
    <>
      <CCard className="mb-4">
        <CCardHeader className="d-flex align-items-center justify-content-between">
          <h5 className="mb-0">Case Id: {apiCase.id}</h5>
          <div>
            <button
              onClick={handleDeleteButtonClick}
              className="btn btn-danger me-1 text-white"
              type="button"
            >
              <span className="btn-icon ">
                <i className="cil-plus"></i>
              </span>
              Delete
            </button>
          </div>
        </CCardHeader>
        <CCardBody>
          <div>
            Created Date: {new Date(apiCase.createdDate * 1000).toLocaleDateString('en-UK')}
          </div>
          <div>Status: {apiCase.status}</div>
          <div>User ID: {apiCase.userId}</div>
          <div>Type: {apiCase.type}</div>
          <div>Vehicle Count: {apiCase.vehicleCount}</div>
          <div>Report Generated: {apiCase.reportGenerated ? 'True' : 'False'}</div>
          <div>Top View Generated: {apiCase.topViewGenerated ? 'True' : 'False'}</div>
          <div>Report URL: {apiCase.reportUrl}</div>
          {apiCase.reportUrlPdf && <div>Email: {apiCase.reportUrlPdf}</div>}
          {apiCase.reportUrlPdfTopVIew && <div>Email: {apiCase.reportUrlPdfTopView}</div>}
          <div>
            Report:
            <pre>{JSON.stringify(apiCase.report, null, 4)}</pre>{' '}
          </div>
          <div>
            Final Dict:
            <pre>{JSON.stringify(apiCase.all_final_dict, null, 4)}</pre>
          </div>
        </CCardBody>
      </CCard>
    </>
  )
}

export default Case
